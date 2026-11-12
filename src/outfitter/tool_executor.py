from __future__ import annotations

import copy
import json
import re
import unicodedata
from typing import Any

from .environment import RENTAL_ACTIVITY_EVENT_TYPES, OutfitterEnvironment
from .errors import (
    ReservationContingencyVersionError,
    ReservationQuoteBatchValidationError,
    ReservationRecoveryBatchValidationError,
)
from .models import MemoryDocument, ToolResult, WorldState
from .tool_specs import tool_specs_for_mode
from .supplier_trial import ACTIVITY_EVENT_TYPES as SUPPLIER_TRIAL_ACTIVITY_EVENT_TYPES
from . import operating_history


class ToolExecutor:
    def __init__(self, environment: OutfitterEnvironment):
        self.environment = environment
        self.tool_interface_version = int(
            environment.dataset.metadata.get("tool_interface_version", 0)
        )
        self.specs = tool_specs_for_mode(
            environment.config.reservation_contract_mode,
            tool_interface_version=self.tool_interface_version,
            assortment_mode=environment.config.assortment_mode,
        )
        self.business_rules_snapshot = (
            environment.business_manual() if environment._v43_actor_manual() else None
        )
        self._query_mode_validators = {}
        if environment._v54_recovery_state():
            from jsonschema import Draft202012Validator
            legacy = tool_specs_for_mode(environment.config.reservation_contract_mode,
                tool_interface_version=51, assortment_mode=environment.config.assortment_mode)
            for spec in legacy:
                if spec.name not in {'view_reservation', 'view_reservation_quote_workspace'}:
                    continue
                discriminator = 'view' if spec.name == 'view_reservation' else 'mode'
                self._query_mode_validators[spec.name] = (discriminator, {
                    branch['properties'][discriminator]['enum'][0]: Draft202012Validator(branch)
                    for branch in spec.parameters['oneOf']})

    def definitions(self) -> list[dict[str, Any]]:
        definitions = [copy.deepcopy(spec.schema()) for spec in self.specs]
        for definition in definitions:
            function = definition["function"]
            properties = function["parameters"].get("properties", {})
            if function["name"] == "business_history":
                function["description"] = (
                    "View completed-day operating history. Summary queries default to the "
                    "last 30 completed days and totals; period may select 7 days, 30 days, "
                    "or season-to-date. Totals cover the full selected period; daily output "
                    "returns at most the latest 30 rows. To inspect successful manager actions for one current or "
                    "completed day, use view=actions with day and no period arguments."
                )
            elif function["name"] == "finance_history":
                function["description"] = (
                    "View factual financial history for completed days. Results default "
                    "to last-30-day totals; period may select 7 days, 30 days, or "
                    "season-to-date. Totals cover the full period and daily output is capped "
                    "at the latest 30 rows."
                )
            elif function["name"] == "service_history":
                function["description"] = (
                    "View quote, booking, and "
                    + (
                        "POS"
                        if self.environment.config.retail_enabled
                        else "walk-in"
                    )
                    + " results for completed days at one "
                    "explicit grain: totals (default), daily, or exact offer. Choose "
                    "last_7_days, last_30_days, or season_to_date; the default is the last "
                    "30 completed days. Totals cover the full period and daily output is "
                    "capped at the latest 30 rows. Exact-offer results are paginated and "
                    "may be filtered with stable offer IDs."
                )
            elif function["name"] == "search_catalog":
                function["description"] = (
                    "Browse a compact kind summary or exact offers in the approved supplier "
                    "catalog. Natural-language search covers names, raw titles, exact SKU "
                    "variants, brands, sellers, and visible specifications."
                )
            elif function["name"] == "catalog_report":
                function["description"] = (
                    "Filter the frozen supplier catalog by natural text, kind, or neutral "
                    "visible specifications. Numeric specifications support lte/gte and "
                    "set-valued specifications support contains. Results are paginated. "
                    "This tool has no view argument; use search_catalog with view=index "
                    "for the compact catalog summary."
                )
            elif function["name"] == "view_rental_activity":
                function["description"] = (
                    "Drill into recorded rental, return, cleaning, repair, retirement, "
                    "acquisition, and sale events. Results are newest first and paginated; "
                    "pass next_cursor as cursor to continue the same filtered query."
                )
        if self.environment._v43_actor_manual():
            from .actor_tool_text import DESCRIPTIONS, V45_DESCRIPTIONS
            for definition in definitions:
                function = definition["function"]
                name = function["name"]
                description = DESCRIPTIONS[name]
                if self.environment._v45_reservation_difficulty():
                    description = V45_DESCRIPTIONS.get(name, description)
                    if name == "set_prices" and self.environment.config.retail_enabled:
                        description += " Retail prices are per consumer unit."
                elif name == "set_prices" and not self.environment.config.retail_enabled:
                    description = ("Set positive posted walk-in prices per asset per day for exact "
                                   "offers. This does not change reservation quotations or contracts.")
                function["description"] = description + (
                    " Read-only; may be batched with independent read-only requests."
                    if self.is_read_only(name) and self.environment.config.parallel_tool_calls
                    else " Read-only; submit one tool per response."
                    if self.is_read_only(name)
                    else " Submit this tool alone in the response."
                )
        if self.environment._v45_reservation_difficulty():
            def describe_prices(value):
                if isinstance(value, dict):
                    for key, child in value.items():
                        if key == "daily_unit_price" and isinstance(child, dict):
                            child["description"] = (
                                "Positive cent-exact first-day rental price per asset. "
                                "Each additional day costs half, rounded half-up to cents per asset."
                            )
                        describe_prices(child)
                elif isinstance(value, list):
                    for child in value:
                        describe_prices(child)
            describe_prices(definitions)
        if self.environment._v47_autonomous_rentals():
            for row in definitions:
                f=row["function"]
                if f["name"]=="respond_reservation_changes":
                    f["description"]="Respond to pre-pickup rescheduling requests only. Customers' continued use is autonomous and cannot be approved or vetoed."
                elif f["name"]=="view_active_rentals":
                    f["description"]="Read active rentals, grouped rental episodes and today's return updates. Contractual dates are distinct from optional, non-guaranteed customer reports. Silence is not confirmation of an on-time return."
        if self.tool_interface_version >= 44:
            for row in definitions:
                f = row["function"]
                if f["name"] == "resolve_reservation_risks":
                    f["description"] += (
                        " All actions form one atomic validation batch: invalid actions undo all business changes."
                        " Complete-plan recovery and line fallbacks cannot be mixed. Genuine payment failure"
                        " or bankruptcy is committed and remaining actions are skipped.")
                elif f["name"] == "view_rental_assets":
                    f["description"] += (
                        " In summary view, status selects SKUs with matching assets; each SKU shows its full"
                        " status distribution. matched_assets counts matches; total_assets includes exited"
                        " historical records. held_assets and owned_assets/supplier_trial_assets exclude"
                        " sold, retired and supplier-returned assets. Asset view still filters individual assets.")
        if self.environment._v48_service_guarantee():
            for row in definitions:
                f=row['function']
                if f['name']=='view_rental_assets':
                    f['description'] += ' Asset view includes repair/maintenance eligibility, reasons, exact fees, duration and conditional completion day; window_open alone is not eligibility.'
                elif f['name']=='resolve_reservation_risks':
                    f['description'] += ' Historical preauthorization does not skip inquire_fallback. Follow recovery_authorization.available_actions.'
                    if self.environment._v51_recovery_integrity():
                        f['description'] += (' Each inquiry must be the only action in its call. Read the customer response before selecting in a later call. '
                            'Complete-plan recovery must retain every currently committed line; historical omit options cannot waive a committed item. '
                            'Use the explicit partial-fulfillment workflow for missing optional items, with rent credit and compensation at actual delivery.')
                    else:
                        f['description'] += ' Ordered inquiry then selection in a batch is allowed.'
                elif f['name']=='cancel_reservation_commitment':
                    f['description'] += ' Compensation uses the original locked first-day guarantee, discounted for notice; same-day cancellation and pickup failure have equal terms. Unearned prepayment refund is separate.'
        if self.environment._v51_recovery_integrity():
            for row in definitions:
                f = row['function']
                if f['name'] == 'view_liquidation_quotes':
                    f['description'] += (' Ineligible or unknown holdings are returned individually in unavailable_items; '
                        'other eligible items still receive quotes. Totals cover only eligible items. This does not relax sale validation.')
                parameters = f['parameters']
                if f['name'] == 'purchase_order_status':
                    parameters['properties']['order_id'].update({
                        'type': ['string', 'null'],
                        'description': 'Omit, null or blank for list mode. A nonblank ID must be the only argument.',
                    })
                    list_fields = ['status', 'placed_day_from', 'placed_day_to', 'limit']
                    parameters.setdefault('allOf', []).append({'anyOf': [
                        {'properties': {'order_id': {'anyOf': [
                            {'type': 'null'}, {'type': 'string', 'pattern': r'^\s*$'}]}}},
                        {'not': {'anyOf': [{'required': [key]} for key in list_fields]}},
                    ]})
                    f['description'] += (
                        ' Two query modes: list orders with status/date/limit filters, omitting order_id; '
                        'or get one exact order using only order_id, with no filters or limit. '
                        'Null, empty or whitespace-only order_id means unspecified (list mode). '
                        'Examples: {"status":"in_transit","limit":20} or {"order_id":"po-00001"}.')
                elif f['name'] == 'search_catalog':
                    parameters['properties']['query'].update({
                        'type': ['string', 'null'],
                        'description': 'Text search in offers mode. In index mode omit it or supply only null/blank.',
                    })
                    parameters.setdefault('allOf', []).append({'anyOf': [
                        {'not': {'required': ['view'], 'properties': {'view': {'const': 'index'}}}},
                        {'properties': {'query': {'anyOf': [
                            {'type': 'null'}, {'type': 'string', 'pattern': r'^\s*$'}]}}},
                    ]})
                    f['description'] += (
                        ' Use view=index for a category summary; query must be omitted, null or blank. '
                        'Use view=offers for text search or paginated offers; blank query matches all. '
                        'Without view, supplying query selects offers, otherwise index. '
                        'Limit/cursor apply only to offers and are ignored by index. '
                        'Examples: {"view":"index"} or {"view":"offers","query":"sleeping bag","limit":10}.')
                if f['name'] == 'resolve_reservation_risks':
                    parameters['properties']['actions'].setdefault('allOf', []).append({
                        'anyOf': [
                            {'maxItems': 1},
                            {'items': {'properties': {'action': {'not': {'enum': [
                                'inquire_fallback', 'inquire_partial_fulfillment']}}}}},
                        ],
                    })
                properties = parameters.get('properties', {})
                if 'period' in properties and 'horizon_days' in properties:
                    parameters.setdefault('allOf', []).append({'not': {'required': ['period', 'horizon_days']}})
                    f['description'] += ' Use either period or horizon_days, never both; omit both for the default window.'
        if operating_history.enabled(self.environment.dataset.metadata):
            from .actor_contract_v53 import tool_description
            for definition in definitions:
                function = definition["function"]
                if function["name"] in operating_history.TOOL_DESCRIPTIONS:
                    function["description"] = operating_history.TOOL_DESCRIPTIONS[function["name"]] + " Read-only."
                if function["name"] == "counter_log":
                    function["parameters"]["properties"]["service_keys"]["items"]["enum"] = sorted({
                        self.environment.service_key_for_product_type(offer.service_line)
                        for offer in self.environment.offers.values()})
                function["description"] = tool_description(function["name"], function["description"],
                                                            read_only=self.is_read_only(function["name"]))
        return definitions

    def effect_map(self) -> dict[str, str]:
        return {spec.name: spec.effect for spec in self.specs}

    def read_only_names(self) -> list[str]:
        return sorted(spec.name for spec in self.specs if spec.effect == "read_only")

    def is_read_only(self, name: str) -> bool:
        return any(
            spec.name == name and spec.effect == "read_only"
            for spec in self.specs
        )

    @staticmethod
    def encode_result(result: ToolResult) -> str:
        return json.dumps({"ok": result.ok, **result.data}, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _object_array(
        args: dict[str, Any], key: str, *, required: bool = True
    ) -> list[dict[str, Any]]:
        if key not in args:
            if required:
                raise ValueError(f"{key} is required")
            return []
        raw = args[key]
        if not isinstance(raw, list):
            raise TypeError(f"{key} must be an array of objects")
        if any(not isinstance(row, dict) for row in raw):
            raise TypeError(f"every item in {key} must be an object")
        return [dict(row) for row in raw]

    @staticmethod
    def _string_array(
        args: dict[str, Any], key: str, *, required: bool = False
    ) -> list[str]:
        if key not in args:
            if required:
                raise ValueError(f"{key} is required")
            return []
        raw = args[key]
        if not isinstance(raw, list):
            raise TypeError(f"{key} must be an array of strings")
        if any(not isinstance(row, str) for row in raw):
            raise TypeError(f"every item in {key} must be a string")
        return list(raw)

    @staticmethod
    def _integer_array(
        args: dict[str, Any], key: str, *, required: bool = False
    ) -> list[int]:
        if key not in args:
            if required:
                raise ValueError(f"{key} is required")
            return []
        raw = args[key]
        if not isinstance(raw, list):
            raise TypeError(f"{key} must be an array of integers")
        if any(isinstance(row, bool) or not isinstance(row, int) for row in raw):
            raise TypeError(f"every item in {key} must be an integer")
        return list(raw)

    @staticmethod
    def _history_horizon(
        state: WorldState,
        args: dict[str, Any],
        *,
        allow_legacy_horizon: bool = True,
    ) -> int:
        if "period" in args and "horizon_days" in args:
            raise ValueError("period and horizon_days cannot be used together")
        if "horizon_days" in args:
            if not allow_legacy_horizon:
                raise ValueError("horizon_days is not valid for this view")
            horizon = int(args["horizon_days"])
            if not 1 <= horizon <= 30:
                raise ValueError("horizon_days must be in [1, 30]")
            return horizon
        period = str(args.get("period", "last_30_days"))
        if period == "last_7_days":
            return 7
        if period == "last_30_days":
            return 30
        if period == "season_to_date":
            return max(1, len(state.daily_history))
        raise ValueError(
            "period must be last_7_days, last_30_days, or season_to_date"
        )

    def execute(self, state: WorldState, name: str, args: dict[str, Any]) -> ToolResult:
        self.environment._assert_v48_state(state)
        try:
            self.environment._v47_assert_state_version(state)
            if name in self._query_mode_validators:
                discriminator, validators = self._query_mode_validators[name]
                mode = args.get(discriminator)
                if not isinstance(mode, str) or mode not in validators:
                    return ToolResult(False, {'error': f'{discriminator} must select a supported query mode',
                        'valid_modes': sorted(validators), 'tool': name})
                error = next(validators[mode].iter_errors(args), None)
                if error is not None:
                    field = '.'.join(map(str, error.absolute_path)) or 'arguments'
                    data = {'error': f'invalid arguments: {field} violates {error.validator}',
                            'field': field, 'constraint': error.validator,
                            'expected': error.validator_value, 'actual_type': type(error.instance).__name__,
                            'tool': name}
                    if error.validator == 'type' and error.validator_value == 'array':
                        data['example'] = ['reservation-001'] if field == 'identifiers' else ['example-id']
                    return ToolResult(False, data)
            result = self._execute(state, name, args)
            result.data = self.environment.actor_assortment_projection(result.data)
            return result
        except ReservationRecoveryBatchValidationError as error:
            return ToolResult(False, {"error": f"invalid arguments: {error}", "tool": name,
                                      "batch_atomic": True, "state_unchanged": True,
                                      **getattr(error,'context',{})})
        except ReservationQuoteBatchValidationError as error:
            return ToolResult(
                False,
                {
                    "error": "reservation quote batch contains invalid items",
                    "tool": name,
                    "invalid_item_count": len(error.invalid_items),
                    "invalid_items": error.invalid_items,
                },
            )
        except ReservationContingencyVersionError as error:
            return ToolResult(
                False,
                {
                    "error": "commitment_version_mismatch",
                    "tool": name,
                    "commitment_id": error.commitment_id,
                    "received_version": error.received_version,
                    "current_version": error.current_version,
                    "state_unchanged": True,
                },
            )
        except (KeyError, TypeError, ValueError) as error:
            return ToolResult(False, {"error": f"invalid arguments: {error}", "tool": name})

    def _execute(self, state: WorldState, name: str, args: dict[str, Any]) -> ToolResult:
        env = self.environment
        env._assert_warehouse_snapshot(state)
        env._assert_durable_acquisition_budget(state)
        if name == "read_business_rules":
            from .actor_manual import TOPICS
            if self.business_rules_snapshot is None:
                return ToolResult(False, {"error": "unknown tool: read_business_rules"})
            if set(args) - {"topics"}:
                raise ValueError("only topics may be supplied")
            if "topics" not in args:
                return ToolResult(True, {"topics": [
                    {"topic": topic, "title": self.business_rules_snapshot[topic]["title"]}
                    for topic in TOPICS
                ]})
            topics = self._string_array(args, "topics", required=True)
            if not topics or len(topics) != len(set(topics)) or any(
                name not in self.business_rules_snapshot for name in topics
            ):
                raise ValueError("topics must contain distinct names from the rule directory")
            return ToolResult(True, {"chapters": [
                {"topic": name, **copy.deepcopy(self.business_rules_snapshot[name])}
                for name in topics
            ]})
        if not env.config.retail_enabled and name == "view_retail_inventory":
            return ToolResult(False, {"error": f"unknown tool: {name}"})
        if name == "view_warehouse_capacity":
            if set(args) - {"offer_ids", "cursor", "limit"}:
                raise ValueError("unexpected warehouse query arguments")
            return ToolResult(True, env.view_warehouse_capacity(state,
                offer_ids=self._string_array(args, "offer_ids"),
                cursor=args.get("cursor", 0), limit=args.get("limit", 50)))
        if name == "business_status":
            return ToolResult(True, env.observation(state, consume_inbox=False))
        if name == "business_history":
            allowed = {"view", "period", "horizon_days", "end_day", "granularity", "day"}
            unexpected = sorted(set(args) - allowed)
            if unexpected:
                raise ValueError(f"unexpected arguments: {unexpected}")
            view = str(args.get("view", "summary"))
            if view == "actions":
                invalid = sorted(set(args) - {"view", "day"})
                if invalid:
                    raise ValueError(
                        f"actions view cannot be combined with period arguments: {invalid}"
                    )
                if "day" not in args:
                    raise ValueError("day is required for the actions view")
                return ToolResult(True, env.business_actions(state, int(args["day"])))
            if view != "summary":
                raise ValueError("view must be summary or actions")
            if "day" in args:
                raise ValueError("day is only valid for the actions view")
            horizon = self._history_horizon(state, args)
            end_day = args.get("end_day")
            if end_day is not None and int(end_day) < 1:
                raise ValueError("end_day must be positive")
            granularity = str(args.get("granularity", "totals"))
            if granularity not in {"totals", "daily"}:
                raise ValueError("granularity must be totals or daily")
            return ToolResult(
                True,
                env.business_history(
                    state,
                    horizon,
                    end_day=None if end_day is None else int(end_day),
                    granularity=granularity,
                ),
            )
        if name == "finance_history":
            allowed = {"period", "horizon_days", "end_day", "granularity"}
            unexpected = sorted(set(args) - allowed)
            if unexpected:
                raise ValueError(f"unexpected arguments: {unexpected}")
            horizon = self._history_horizon(state, args)
            end_day = args.get("end_day")
            if end_day is not None and int(end_day) < 1:
                raise ValueError("end_day must be positive")
            granularity = str(args.get("granularity", "totals"))
            if granularity not in {"totals", "daily"}:
                raise ValueError("granularity must be totals or daily")
            return ToolResult(
                True,
                env.finance_history(
                    state,
                    horizon,
                    end_day=None if end_day is None else int(end_day),
                    granularity=granularity,
                ),
            )
        if name == "service_history":
            allowed = {
                "period",
                "horizon_days",
                "end_day",
                "granularity",
                "offer_ids",
                "cursor",
                "limit",
            }
            if operating_history.enabled(env.dataset.metadata):
                allowed.add("channel")
            unexpected = sorted(set(args) - allowed)
            if unexpected:
                raise ValueError(f"unexpected arguments: {unexpected}")
            horizon = self._history_horizon(state, args)
            end_day = args.get("end_day")
            if end_day is not None and int(end_day) < 1:
                raise ValueError("end_day must be positive")
            granularity = str(args.get("granularity", "totals"))
            valid_granularities = {"totals", "daily", "offer"}
            if granularity not in valid_granularities:
                raise ValueError(
                    "granularity must be totals, daily, or offer"
                )
            offer_ids = list(map(str, args.get("offer_ids", [])))
            if len(offer_ids) > 20 or len(set(offer_ids)) != len(offer_ids):
                raise ValueError("offer_ids must contain at most 20 unique values")
            unknown_offers = sorted(set(offer_ids) - set(env.offers))
            if unknown_offers:
                return ToolResult(
                    False,
                    {"error": f"unknown offer_ids: {unknown_offers}"},
                )
            if offer_ids and granularity != "offer":
                raise ValueError("offer_ids are only valid for offer granularity")
            cursor = int(args.get("cursor", 0))
            limit = int(args.get("limit", 20))
            if (cursor or "limit" in args) and granularity != "offer":
                raise ValueError(
                    "cursor and limit are only valid for offer granularity"
                )
            if cursor < 0 or not 1 <= limit <= 50:
                raise ValueError(
                    "cursor must be non-negative and limit must be in [1, 50]"
                )
            return ToolResult(
                True,
                env.service_history(
                    state,
                    horizon,
                    offer_ids=offer_ids or None,
                    end_day=None if end_day is None else int(end_day),
                    granularity=granularity,
                    cursor=cursor,
                    limit=limit,
                    channel=str(args.get("channel", "all")),
                ),
            )
        if name == "counter_log":
            allowed = {"day", "cursor", "limit"}
            if operating_history.enabled(env.dataset.metadata):
                allowed.update({"view", "from_day", "to_day", "offer_ids", "service_keys"})
            unexpected = sorted(set(args) - allowed)
            if unexpected:
                raise ValueError(f"unexpected arguments: {unexpected}")
            day = args.get("day")
            cursor = int(args.get("cursor", 0))
            limit = int(args.get("limit", 10))
            if cursor < 0 or not 1 <= limit <= 50:
                raise ValueError(
                    "cursor must be non-negative and limit must be in [1, 50]"
                )
            return ToolResult(
                True,
                env.counter_log(
                    state,
                    day=None if day is None else int(day),
                    cursor=cursor,
                    limit=limit,
                    **{key: args[key] for key in ("view", "from_day", "to_day", "offer_ids", "service_keys") if key in args},
                ),
            )
        if name == "demand_report":
            allowed = {
                "group_by",
                "filters",
                "last_n_days",
                "from_day",
                "to_day",
                "cursor",
                "limit",
            }
            unexpected = sorted(set(args) - allowed)
            if unexpected:
                raise ValueError(f"unexpected arguments: {unexpected}")
            group_by = list(map(str, args["group_by"]))
            if not 1 <= len(group_by) <= 3 or len(set(group_by)) != len(group_by):
                raise ValueError("group_by must contain 1-3 unique dimensions")
            if not env.config.retail_enabled and (
                "rental_or_retail" in group_by
                or "rental_or_retail" in dict(args.get("filters") or {})
            ):
                raise ValueError("the requested demand dimension is not available")
            if "last_n_days" in args and ({"from_day", "to_day"} & set(args)):
                raise ValueError("last_n_days cannot be combined with from_day/to_day")
            last_completed = int(state.daily_history[-1]["day"]) if state.daily_history else 0
            if "last_n_days" in args:
                count = int(args["last_n_days"])
                start_day = max(1, last_completed - count + 1)
                end_day = last_completed
            else:
                start_day = int(args.get("from_day", max(1, last_completed - 29)))
                end_day = int(args.get("to_day", last_completed))
            raw_filters = args.get("filters") or {}
            if not isinstance(raw_filters, dict):
                raise ValueError("filters must be an object")
            filters = {
                str(key): list(value) for key, value in raw_filters.items()
            }
            cursor = int(args.get("cursor", 0))
            limit = int(args.get("limit", 20))
            if cursor < 0 or not 1 <= limit <= 50:
                raise ValueError(
                    "cursor must be non-negative and limit must be in [1, 50]"
                )
            empty_opening_window = (
                last_completed == 0
                and "from_day" not in args
                and "to_day" not in args
            )
            if not empty_opening_window and (
                start_day < 1 or end_day < start_day or end_day > last_completed
            ):
                raise ValueError(
                    f"window must be within completed days 1..{last_completed}"
                )
            return ToolResult(
                True,
                env.demand_report(
                    state,
                    group_by=group_by,
                    filters=filters,
                    start_day=start_day,
                    end_day=end_day,
                    cursor=cursor,
                    limit=limit,
                ),
            )
        if name == "catalog_report":
            if "view" in args:
                raise ValueError(
                    "catalog_report has no view argument; use search_catalog "
                    "with view=index for the compact catalog summary"
                )
            allowed = {
                "filters",
                "query",
                "specification_filters",
                "offset",
                "cursor",
                "limit",
            }
            if env.config.retail_enabled:
                allowed.add("kind")
            unexpected = sorted(set(args) - allowed)
            if unexpected:
                raise ValueError(f"unexpected arguments: {unexpected}")
            if "offset" in args and "cursor" in args:
                raise ValueError("use cursor or offset, not both")
            offset = int(args.get("cursor", args.get("offset", 0)))
            limit = int(args.get("limit", 10))
            if offset < 0 or not 1 <= limit <= 20:
                raise ValueError("offset must be non-negative and limit must be in [1, 20]")
            filters = dict(args.get("filters") or {})
            if not env.config.retail_enabled and "kind" in filters:
                raise ValueError("unexpected catalog filters: ['kind']")
            for alias in ("query", "kind"):
                if alias not in args:
                    continue
                if alias in filters and filters[alias] != args[alias]:
                    raise ValueError(
                        f"conflicting top-level and filters.{alias} values"
                    )
                filters[alias] = args[alias]
            payload = env.catalog_report(
                filters=filters,
                attribute_filters=self._object_array(
                    args, "specification_filters", required=False
                ),
                offset=offset,
                limit=limit,
            )
            for row in payload["offers"]:
                offer_id = str(row["offer_id"])
                row["current_price"] = state.prices.get(offer_id)
                row["priced_for_sale"] = offer_id in state.prices
                if offer_id in state.deposits:
                    row["fixed_deposit"] = state.deposits[offer_id]
                    row["accepting_new_rentals"] = state.rental_service_open[
                        offer_id
                    ]
            return ToolResult(True, payload)
        if name == "weather_forecast":
            horizon = int(args.get("horizon_days", 7))
            maximum_horizon = 15 if self.tool_interface_version >= 34 else 14
            if not 1 <= horizon <= maximum_horizon:
                raise ValueError(
                    f"horizon_days must be in [1, {maximum_horizon}]"
                )
            return ToolResult(True, {"forecast": env.weather_forecast(state, horizon)})
        if name == "search_catalog":
            if env._v51_recovery_integrity() and 'query' in args:
                query = args['query']
                if query is not None and not isinstance(query, str):
                    raise ValueError('query must be a string or null')
                args = dict(args)
                args['query'] = '' if query is None else query.strip()
                if args.get('view') == 'index' and not args['query']:
                    args.pop('query')
            allowed = {"view", "query", "limit", "cursor"}
            if env.config.retail_enabled:
                allowed.add("kind")
            unexpected = sorted(set(args) - allowed)
            if unexpected:
                raise ValueError(f"unexpected arguments: {unexpected}")
            view = str(
                args.get(
                    "view",
                    "offers" if "query" in args else "index",
                )
            )
            public_kind = args.get("kind")
            kind_map = {"rental_asset": "durable", "retail_item": "consumable"}
            if public_kind is not None and str(public_kind) not in kind_map:
                raise ValueError("kind must be rental_asset or retail_item")
            asset_kind = None if public_kind is None else kind_map[str(public_kind)]
            if view == "index":
                if "query" in args:
                    if env._v51_recovery_integrity():
                        raise ValueError(
                            'nonblank query requires view="offers"; '
                            'use {"view":"index"} for a category summary'
                        )
                    raise ValueError(
                        "query is only valid for the offers view"
                    )
                return ToolResult(
                    True,
                    {
                        "view": view,
                        "kind_summary": env.catalog_index(
                            asset_kind
                        ),
                    },
                )
            if view != "offers":
                raise ValueError("view must be index or offers")
            query = str(args.get("query", "")).strip()
            limit = int(args.get("limit", 10))
            cursor = int(args.get("cursor", 0))
            if cursor < 0 or not 1 <= limit <= 20:
                raise ValueError(
                    "cursor must be non-negative and limit must be in [1, 20]"
                )
            payload = env.search_catalog(
                query, [], asset_kind, limit, cursor=cursor
            )
            for row in payload["offers"]:
                offer_id = str(row["offer_id"])
                row["current_price"] = state.prices.get(offer_id)
                row["priced_for_sale"] = offer_id in state.prices
                if offer_id in state.deposits:
                    row["fixed_deposit"] = state.deposits[offer_id]
                    row["accepting_new_rentals"] = state.rental_service_open[
                        offer_id
                    ]
            return ToolResult(True, {"view": view, **payload})
        if name == "inspect_offer":
            offer_id = str(args["offer_id"])
            if offer_id not in env.offers:
                return ToolResult(False, {"error": f"unknown offer_id: {offer_id}"})
            result = env.public_offer(offer_id, detailed=True)
            result.update(
                current_price=state.prices.get(offer_id),
                priced_for_sale=offer_id in state.prices,
            )
            if offer_id in state.deposits:
                result["fixed_deposit"] = state.deposits[offer_id]
                result["accepting_new_rentals"] = state.rental_service_open[offer_id]
            return ToolResult(True, {"offer": result})
        if name == "place_purchase_order":
            return env.place_purchase_order(
                state,
                str(args["seller_id"]),
                self._object_array(args, "items"),
                str(args.get("procurement_mode", "owned")),
            )
        if name == "return_supplier_trial_assets":
            return env.return_supplier_trial_assets(
                state, self._string_array(args, "asset_ids", required=True)
            )
        if name == "purchase_order_status":
            if env._v51_recovery_integrity():
                allowed = {'order_id', 'status', 'placed_day_from', 'placed_day_to', 'limit'}
                if set(args) - allowed:
                    raise ValueError(f'unexpected arguments: {sorted(set(args) - allowed)}')
                raw_order_id = args.get('order_id')
                if raw_order_id is not None and not isinstance(raw_order_id, str):
                    raise ValueError('order_id must be a string or null')
                if raw_order_id is None or not raw_order_id.strip():
                    args = {key: value for key, value in args.items() if key != 'order_id'}
            order_id = args.get("order_id")
            if order_id is not None:
                if set(args) != {"order_id"}:
                    if env._v51_recovery_integrity():
                        raise ValueError(
                            'nonblank order_id must be the only argument; '
                            'use {"order_id":"po-00001"} for one order, or omit order_id '
                            'and use {"status":"in_transit","limit":20} for a list'
                        )
                    raise ValueError("order_id cannot be combined with list filters")
                order = next(
                    (row for row in state.orders if row.order_id == str(order_id)),
                    None,
                )
                if order is None:
                    return ToolResult(False, {"error": f"unknown order_id: {order_id}"})
                return ToolResult(True, {
                    "order": env._order_view(order),
                    **({"warehouse_capacity": env._warehouse_summary(state)} if env._v31_warehouse_capacity() else {}),
                    **({"delivery_timing": env._order_delivery_timing()}
                       if env._v29_reservation_accountability() else {}),
                })
            status = str(args.get("status", "in_transit"))
            if status not in {"in_transit", "delivered"}:
                raise ValueError("status must be in_transit or delivered")
            day_from = int(args.get("placed_day_from", 1))
            day_to = int(args.get("placed_day_to", state.max_days))
            limit = int(args.get("limit", 20))
            if day_from > day_to:
                raise ValueError("placed_day_from must not exceed placed_day_to")
            if not 1 <= limit <= 50:
                raise ValueError("limit must be in [1, 50]")
            matches = [
                row
                for row in state.orders
                if row.status == status and day_from <= row.placed_day <= day_to
            ]
            matches.sort(
                key=(
                    (
                        lambda row: (
                            env.estimated_order_delivery_day(row),
                            row.order_id,
                        )
                    )
                    if status == "in_transit"
                    else (lambda row: (-row.placed_day, row.order_id))
                )
            )
            rows = [env._order_view(row) for row in matches[:limit]]
            return ToolResult(
                True,
                {
                    "status": status,
                    **({"warehouse_capacity": env._warehouse_summary(state)} if env._v31_warehouse_capacity() else {}),
                    **({"delivery_timing": env._order_delivery_timing()}
                       if env._v29_reservation_accountability() else {}),
                    "summary": {
                        "matching_order_count": len(matches),
                        "returned_order_count": len(rows),
                        "prepaid_total": round(sum(row.total_cost for row in matches), 2),
                        "next_estimated_delivery_day": (
                            min(
                                env.estimated_order_delivery_day(row)
                                for row in matches
                            )
                            if status == "in_transit" and matches
                            else None
                        ),
                    },
                    "orders": rows,
                },
            )
        if name == "view_retail_inventory":
            offer_ids = list(map(str, args.get("offer_ids", [])))
            if len(offer_ids) > 20 or len(set(offer_ids)) != len(offer_ids):
                raise ValueError("offer_ids must contain at most 20 unique values")
            retail_offer_ids = {
                offer.offer_id
                for offer in env.dataset.offers
                if offer.asset_kind == "consumable"
            }
            unknown = sorted(set(offer_ids) - retail_offer_ids)
            if unknown:
                return ToolResult(
                    False,
                    {"error": f"unknown retail offer_ids: {unknown}"},
                )
            selected = sorted(set(offer_ids) or retail_offer_ids)

            transit_by_offer: dict[str, dict[str, Any]] = {}
            for order in state.orders:
                if order.status != "in_transit":
                    continue
                for item in order.items:
                    offer_id = str(item["offer_id"])
                    row = transit_by_offer.setdefault(
                        offer_id, {"units": 0, "estimated_delivery_days": []}
                    )
                    row["units"] += int(item["packs"] * item["package_units"])
                    if env._v31_warehouse_capacity():
                        row["units"] -= item["received_units"]
                    row["estimated_delivery_days"].append(
                        env.estimated_order_delivery_day(order)
                    )

            rows = []
            for offer_id in selected:
                offer = env.offers[offer_id]
                inventory = state.retail_inventory.get(offer_id, {})
                quantity = int(inventory.get("quantity", 0))
                transit = transit_by_offer.get(
                    offer_id, {"units": 0, "estimated_delivery_days": []}
                )
                if not offer_ids and quantity <= 0 and int(transit["units"]) <= 0:
                    continue
                rows.append(
                    {
                        "offer_id": offer_id,
                        "name": offer.name,
                        "on_hand_units": quantity,
                        "average_acquisition_unit_cost": round(
                            float(inventory.get("avg_unit_cost", offer.unit_cost)), 4
                        ),
                        "current_price": state.prices.get(offer_id),
                        "priced_for_sale": offer_id in state.prices,
                        "in_transit_units": int(transit["units"]),
                        "next_estimated_delivery_day": (
                            min(transit["estimated_delivery_days"])
                            if transit["estimated_delivery_days"]
                            else None
                        ),
                    }
                )
            return ToolResult(True, {"offers": rows})
        if name == "view_rental_assets":
            if self.tool_interface_version >= 34 and "view" not in args:
                raise ValueError("V39 view_rental_assets requires an explicit view")
            view = str(args.get("view", "summary"))
            if self.tool_interface_version >= 34:
                allowed = (
                    {"view", "status", "offer_id"}
                    if view == "summary"
                    else {"view", "status", "offer_id", "cursor", "limit"}
                    if view == "asset"
                    else {"view"}
                )
                unexpected = sorted(set(args) - allowed)
                if unexpected:
                    raise ValueError(
                        f"unexpected {view} arguments: {unexpected}"
                    )
            status = args.get("status")
            offer_id = args.get("offer_id")
            matches = [
                asset
                for asset in state.assets
                if (offer_id is None or asset.offer_id == str(offer_id))
                and (status is None or asset.status == str(status))
            ]
            if view == "summary":
                unexpected_pagination = sorted(
                    set(args).intersection({"cursor", "limit"})
                )
                if unexpected_pagination:
                    raise ValueError(
                        "cursor and limit are only valid for the asset view"
                    )
                statuses = (
                    "available",
                    "rented",
                    "cleaning",
                    "damaged",
                    "repairing",
                    "maintenance",
                    "maintenance_hold",
                    "retired",
                    "sold",
                    *(("supplier_returned",) if env._v37_supplier_trial() else ()),
                )
                rows = []
                for grouped_offer_id in sorted({asset.offer_id for asset in matches}):
                    full_summary = self.tool_interface_version >= 44
                    assets = [asset for asset in (state.assets if full_summary else matches)
                              if asset.offer_id == grouped_offer_id]
                    held_assets = ([asset for asset in assets if env._warehouse_asset_held(asset)]
                                   if full_summary else assets)
                    offer = env.offers[grouped_offer_id]
                    rows.append(
                        {
                            "offer_id": grouped_offer_id,
                            "name": offer.name,
                            "total_assets": len(assets),
                            **({"matched_assets": sum(asset.offer_id == grouped_offer_id for asset in matches),
                                "held_assets": len(held_assets)} if full_summary else {}),
                            **{
                                asset_status: sum(
                                    asset.status == asset_status for asset in assets
                                )
                                for asset_status in statuses
                            },
                            "daily_price": state.prices.get(grouped_offer_id),
                            "priced_for_rental": grouped_offer_id in state.prices,
                            "fixed_deposit": state.deposits[grouped_offer_id],
                            "accepting_new_rentals": state.rental_service_open[
                                grouped_offer_id
                            ],
                            "inspection_due_assets": sum(
                                (
                                    bool(
                                        env._lifecycle_maintenance_status(asset)[
                                            "window_open"
                                        ]
                                    )
                                    if env.config.lifecycle_economics_enabled
                                    else asset.next_inspection_due_day is not None
                                    and state.day >= asset.next_inspection_due_day
                                )
                                for asset in held_assets
                            ),
                            **(
                                {
                                    "owned_assets": sum(
                                        asset.ownership_type == "owned"
                                        for asset in held_assets
                                    ),
                                    "supplier_trial_assets": sum(
                                        asset.ownership_type == "supplier_trial"
                                        for asset in held_assets
                                    ),
                                }
                                if env._v37_supplier_trial()
                                else {}
                            ),
                        }
                    )
                return ToolResult(
                    True,
                    {
                        "view": view,
                        "status_filter": status,
                        "asset_groups": rows,
                    },
                )
            if view != "asset":
                raise ValueError("view must be summary or asset")
            cursor = int(args.get("cursor", 0))
            limit = int(args.get("limit", 20))
            if cursor < 0 or not 1 <= limit <= 50:
                raise ValueError(
                    "cursor must be non-negative and limit must be in [1, 50]"
                )
            detail_matches = sorted(matches, key=lambda row: row.asset_id)
            page = detail_matches[cursor : cursor + limit]
            rows = [
                {
                    "asset_id": asset.asset_id,
                    "offer_id": asset.offer_id,
                    "name": env.offers[asset.offer_id].name,
                    **({"holding_costs": operating_history.holding_costs(env, asset.offer_id, asset),
                        "wear_description": "Accumulated usage wear, not next-rental damage/retirement probability; maintenance does not reset it."}
                       if operating_history.enabled(env.dataset.metadata) else {}),
                    "used_life_days": asset.used_life_days,
                    "completed_rental_cycles": asset.used_rental_cycles,
                    "status": asset.status,
                    "ready_day": asset.ready_day,
                    **(
                        {
                            "ownership_type": asset.ownership_type,
                            "supplier_trial": (
                                None
                                if asset.ownership_type != "supplier_trial"
                                else env._public_supplier_trial(
                                    env._supplier_trial_record_for_asset(
                                        state, asset
                                    )
                                )
                            ),
                        }
                        if env._v37_supplier_trial()
                        else {}
                    ),
                    **({"physically_held": env._warehouse_asset_held(asset),
                        "disposed_day": asset.disposed_day, "sold_day": asset.sold_day,
                        "warehouse_capacity_units_per_asset": env._warehouse_weight(asset.offer_id)}
                       if env._v31_warehouse_capacity() else {}),
                    **(
                        {
                            "condition_ratio": round(asset.condition_ratio, 6),
                            "combined_wear": round(
                                env._combined_wear(asset), 6
                            ),
                            "maintenance_wear_anchor": round(
                                asset.maintenance_wear_anchor, 6
                            ),
                            "maintenance_window_open": bool(
                                env._lifecycle_maintenance_status(asset)[
                                    "window_open"
                                ]
                            ),
                            "maintenance_overdue": bool(
                                env._lifecycle_maintenance_status(asset)[
                                    "overdue"
                                ]
                            ),
                            "wear_since_maintenance": env._lifecycle_maintenance_status(
                                asset
                            )["wear_since_maintenance"],
                            "maintenance_due_wear": env._lifecycle_maintenance_status(
                                asset
                            )["maintenance_due_wear"],
                            "hazard_protection": env._lifecycle_maintenance_status(
                                asset
                            )["hazard_protection"],
                        }
                        if env.config.lifecycle_economics_enabled
                        else {
                            "next_inspection_due_day": asset.next_inspection_due_day,
                            "maintenance_window_open": (
                                asset.next_inspection_due_day is not None
                                and state.day
                                >= asset.next_inspection_due_day
                                - int(env.config.maintenance_due_window_days)
                            ),
                        }
                    ),
                    "daily_price": state.prices.get(asset.offer_id),
                    "priced_for_rental": asset.offer_id in state.prices,
                    "fixed_deposit": state.deposits[asset.offer_id],
                    "accepting_new_rentals": state.rental_service_open[
                        asset.offer_id
                    ],
                    **env._public_asset_durable_capital(state, asset),
                    **({'repair':env.asset_service_terms(state,asset,'repair'),
                        'maintenance':env.asset_service_terms(state,asset,'maintenance')}
                       if env._v48_service_guarantee() else {}),
                }
                for asset in page
            ]
            next_cursor = (
                cursor + len(rows)
                if cursor + len(rows) < len(detail_matches)
                else None
            )
            return ToolResult(
                True,
                {
                    "view": view,
                    "assets": rows,
                    "total_count": len(detail_matches),
                    "returned_count": len(rows),
                    "cursor": cursor,
                    "limit": limit,
                    "has_more": next_cursor is not None,
                    "next_cursor": next_cursor,
                },
            )
        if name == "view_active_rentals":
            rows = [
                {
                    "contract_id": row.contract_id,
                    "asset_id": row.asset_id,
                    "offer_id": row.offer_id,
                    "name": env.offers[row.offer_id].name,
                    "start_day": row.start_day,
                    "return_day": row.return_day,
                    "promised_return_day": row.promised_return_day,
                    "approved_return_day": row.approved_return_day,
                    "estimated_return_day": row.estimated_return_day,
                    "actual_return_day": row.actual_return_day,
                    "duration_days": row.duration_days,
                    "rent_paid": row.rent_paid,
                    "refundable_deposit": row.deposit,
                    **(
                        {
                            "supplier_trial_id": row.supplier_trial_id,
                            "supplier_trial_usage_fee": round(
                                row.supplier_trial_usage_fee, 2
                            ),
                        }
                        if env._v37_supplier_trial()
                        and row.supplier_trial_id is not None
                        else {}
                    ),
                }
                for row in state.contracts
                if row.status == "active"
            ]
            return ToolResult(True, env._v47_active_payload(state,rows) if env._v47_autonomous_rentals()
                              else {"rentals": rows})
        if name in {
            "reservation_inbox",
            "view_reservation_calendar",
            "view_rental_capacity",
            "view_reservation_quote_workspace",
            "set_walk_in_capacity_limits",
            "view_reservation",
            "view_reservation_options",
            "quote_reservation_inquiries",
            "confirm_reservation_plans",
            "respond_reservation_service_offers",
            "stage_reservation_line",
            "respond_reservation_changes",
            "resolve_reservation_risks",
            "cancel_reservation_commitment",
            "request_reservation_contingency",
            "apply_reservation_contingency",
        } and not env.config.reservations_enabled:
            return ToolResult(False, {"error": "advance reservations are disabled"})
        if name == "reservation_inbox":
            if args:
                raise ValueError("reservation_inbox takes no arguments")
            return ToolResult(True, env.reservation_inbox(state))
        if name == "view_rental_capacity":
            if self.tool_interface_version < 18:
                return ToolResult(False, {"error": f"unknown tool: {name}"})
            view = str(args.get("view", ""))
            offer_ids = self._string_array(args, "offer_ids", required=False)
            if "offer_ids" in args and not offer_ids:
                raise ValueError("offer_ids must not be empty")
            if len(offer_ids) > 20 or len(set(offer_ids)) != len(offer_ids):
                raise ValueError("offer_ids must contain at most 20 unique values")
            unknown = sorted(set(offer_ids) - set(env.offers))
            if unknown:
                raise ValueError(f"unknown offer_ids: {unknown}")
            non_rental = sorted(
                row
                for row in offer_ids
                if env.offers[row].asset_kind != "durable"
            )
            if non_rental:
                raise ValueError(
                    "rental capacity accepts rental offer_ids only; "
                    f"non-rental offer_ids: {non_rental}"
                )
            if view == "timeline":
                unexpected = sorted(
                    set(args) - {"view", "offer_ids", "from_day", "to_day"}
                )
                if unexpected:
                    raise ValueError(
                        f"unexpected timeline arguments: {unexpected}"
                    )
                from_day = int(args.get("from_day", state.day))
                to_day = int(
                    args.get("to_day", min(state.max_days, state.day + 6))
                )
                if (
                    from_day < 1
                    or to_day < from_day
                    or to_day > state.max_days
                ):
                    raise ValueError(
                        "timeline window must satisfy "
                        f"1 <= from_day <= to_day <= {state.max_days}"
                    )
                maximum_inclusive_dates = (
                    15 if self.tool_interface_version >= 34 else 14
                )
                if to_day - from_day + 1 > maximum_inclusive_dates:
                    maximum_to_day = min(
                        state.max_days, from_day + maximum_inclusive_dates - 1
                    )
                    raise ValueError(
                        "rental capacity timeline uses inclusive dates and "
                        "requires to_day - from_day <= "
                        f"{maximum_inclusive_dates - 1}; "
                        f"for from_day={from_day}, maximum valid to_day is "
                        f"{maximum_to_day}"
                    )
                return ToolResult(
                    True,
                    env.view_rental_capacity(
                        state,
                        view=view,
                        offer_ids=offer_ids,
                        from_day=from_day,
                        to_day=to_day,
                    ),
                )
            if view == "service_jobs":
                unexpected = sorted(
                    set(args)
                    - {"view", "offer_ids", "job_types", "cursor", "limit"}
                )
                if unexpected:
                    raise ValueError(
                        f"unexpected service_jobs arguments: {unexpected}"
                    )
                job_types = self._string_array(
                    args, "job_types", required=False
                )
                if "job_types" in args and not job_types:
                    raise ValueError("job_types must not be empty")
                if len(job_types) != len(set(job_types)) or any(
                    row not in {"cleaning", "maintenance", "repair"}
                    for row in job_types
                ):
                    raise ValueError(
                        "job_types must contain unique cleaning, maintenance, or repair values"
                    )
                cursor = int(args.get("cursor", 0))
                limit = int(args.get("limit", 10))
                if cursor < 0 or not 1 <= limit <= 20:
                    raise ValueError(
                        "cursor must be non-negative and limit must be in [1, 20]"
                    )
                return ToolResult(
                    True,
                    env.view_rental_capacity(
                        state,
                        view=view,
                        offer_ids=offer_ids,
                        job_types=job_types or None,
                        cursor=cursor,
                        limit=limit,
                    ),
                )
            raise ValueError("view must be timeline or service_jobs")
        if name == "view_reservation_calendar":
            if self.tool_interface_version >= 18:
                return ToolResult(False, {"error": f"unknown tool: {name}"})
            offer_ids = self._string_array(args, "offer_ids", required=False)
            unknown = sorted(set(offer_ids) - set(env.offers))
            if unknown:
                raise ValueError(f"unknown offer_ids: {unknown}")
            non_rental = sorted(
                row
                for row in offer_ids
                if env.offers[row].asset_kind != "durable"
            )
            if non_rental:
                raise ValueError(
                    "reservation calendar accepts rental offer_ids only; "
                    f"non-rental offer_ids: {non_rental}"
                )
            from_day = int(args.get("from_day", state.day))
            to_day = int(args.get("to_day", min(state.max_days, state.day + 6)))
            if from_day < state.day or to_day < from_day or to_day > state.max_days:
                raise ValueError(
                    f"calendar window must satisfy {state.day} <= from_day <= to_day <= {state.max_days}"
                )
            if to_day - from_day + 1 > 14:
                raise ValueError("reservation calendar window may span at most 14 days")
            return ToolResult(
                True,
                env.view_reservation_calendar(
                    state,
                    offer_ids=offer_ids,
                    from_day=from_day,
                    to_day=to_day,
                ),
            )
        if name == "view_reservation":
            identifiers = self._string_array(
                args, "identifiers", required=False
            )
            view = str(
                args.get("view", "detail" if identifiers else "index")
            )
            if self.tool_interface_version >= 20:
                if "view" not in args:
                    raise ValueError(
                        "V24 view_reservation requires an explicit view"
                    )
                allowed_by_view = {
                    "index": {"view", "cursor", "limit"},
                    "detail": {"view", "identifiers", "sections"},
                    "timeline": {"view", "identifiers", "cursor", "limit"},
                    "quote_history": {
                        "view",
                        "identifiers",
                        "from_day",
                        "to_day",
                        "offer_ids",
                        "service_keys",
                        "quote_outcomes",
                        "menu_sizes",
                        "authorized_plan_counts",
                        "decline_feedback_tags",
                        "decline_feedback_concern_sets",
                        "sort_order",
                        "cursor",
                        "limit",
                    },
                }
                if view not in allowed_by_view:
                    raise ValueError(
                        "view must be index, detail, timeline, or quote_history"
                    )
                unexpected = sorted(set(args) - allowed_by_view[view])
                if unexpected:
                    raise ValueError(
                        f"unexpected {view} arguments: {unexpected}"
                    )
                if view == "detail" and (
                    not 1 <= len(identifiers) <= 3
                    or len(set(identifiers)) != len(identifiers)
                ):
                    raise ValueError(
                        "detail view requires 1 to 3 unique identifiers"
                    )
                if view == "timeline" and len(identifiers) != 1:
                    raise ValueError(
                        "timeline view requires exactly one identifier"
                    )
            cursor = int(args.get("cursor", 0))
            sections = (
                self._string_array(args, "sections", required=False)
                if "sections" in args
                else None
            )
            if sections == []:
                raise ValueError("sections must not be empty")
            searchable_quote_history = self.tool_interface_version >= 13
            quote_history_arrays: dict[str, list[Any] | None] = {}
            for key in (
                "offer_ids",
                "service_keys",
                "quote_outcomes",
                "assessment_categories",
                "decline_feedback_categories",
                "decline_feedback_tags",
                "decline_feedback_concern_sets",
            ):
                value = (
                    self._string_array(args, key, required=False)
                    if key in args
                    else None
                )
                if value == []:
                    raise ValueError(f"{key} must not be empty")
                quote_history_arrays[key] = value
            menu_sizes = (
                self._integer_array(args, "menu_sizes", required=False)
                if "menu_sizes" in args
                else None
            )
            if menu_sizes == []:
                raise ValueError("menu_sizes must not be empty")
            contract_option_counts = (
                self._integer_array(
                    args, "contract_option_counts", required=False
                )
                if "contract_option_counts" in args
                else None
            )
            if contract_option_counts == []:
                raise ValueError("contract_option_counts must not be empty")
            authorized_plan_counts = (
                self._integer_array(
                    args, "authorized_plan_counts", required=False
                )
                if "authorized_plan_counts" in args
                else None
            )
            if authorized_plan_counts == []:
                raise ValueError("authorized_plan_counts must not be empty")
            limit = int(
                args.get(
                    "limit",
                    10
                    if self.tool_interface_version >= 18
                    or (searchable_quote_history and view == "quote_history")
                    else 20,
                )
            )
            return ToolResult(
                True,
                env.view_reservation(
                    state,
                    identifiers,
                    view=view,
                    cursor=cursor,
                    limit=limit,
                    from_day=(
                        None
                        if args.get("from_day") is None
                        else int(args["from_day"])
                    ),
                    to_day=(
                        None
                        if args.get("to_day") is None
                        else int(args["to_day"])
                    ),
                    offer_ids=quote_history_arrays["offer_ids"],
                    service_keys=quote_history_arrays["service_keys"],
                    quote_outcomes=quote_history_arrays["quote_outcomes"],
                    menu_sizes=menu_sizes,
                    contract_option_counts=contract_option_counts,
                    assessment_categories=quote_history_arrays[
                        "assessment_categories"
                    ],
                    authorized_plan_counts=authorized_plan_counts,
                    decline_feedback_categories=quote_history_arrays[
                        "decline_feedback_categories"
                    ],
                    decline_feedback_tags=quote_history_arrays[
                        "decline_feedback_tags"
                    ],
                    decline_feedback_concern_sets=quote_history_arrays[
                        "decline_feedback_concern_sets"
                    ],
                    sections=sections,
                    sort_order=(
                        None
                        if args.get("sort_order") is None
                        else str(args["sort_order"])
                    ),
                    searchable_quote_history=searchable_quote_history,
                ),
            )
        if name == "view_reservation_quote_workspace":
            if (
                self.tool_interface_version < 21
                or env.config.reservation_contract_mode
                != "factorized_line_options_plan_set_v1"
            ):
                return ToolResult(False, {"error": f"unknown tool: {name}"})
            mode = str(args.get("mode", ""))
            if mode == "pending_page":
                unexpected = sorted(
                    set(args)
                    - {"mode", "cursor", "limit", "offer_projection"}
                )
                if unexpected:
                    raise ValueError(
                        f"unexpected pending_page arguments: {unexpected}"
                    )
                request_ids = None
                raw_cursor = args.get("cursor")
                if raw_cursor is not None and not isinstance(raw_cursor, str):
                    raise TypeError("cursor must be a request ID string")
                cursor = raw_cursor
                raw_limit = args.get("limit", 5)
                if isinstance(raw_limit, bool) or not isinstance(raw_limit, int):
                    raise TypeError("limit must be an integer")
                limit = raw_limit
            elif mode == "requests":
                unexpected = sorted(
                    set(args) - {"mode", "request_ids", "offer_projection"}
                )
                if unexpected:
                    raise ValueError(
                        f"unexpected requests arguments: {unexpected}"
                    )
                request_ids = self._string_array(
                    args, "request_ids", required=True
                )
                cursor = None
                limit = len(request_ids)
            else:
                raise ValueError("mode must be pending_page or requests")
            offer_projection = args.get("offer_projection", "compact")
            if not isinstance(offer_projection, str):
                raise TypeError("offer_projection must be a string")
            return ToolResult(
                True,
                env.view_reservation_quote_workspace(
                    state,
                    mode=mode,
                    request_ids=request_ids,
                    cursor=cursor,
                    limit=limit,
                    offer_projection=offer_projection,
                ),
            )
        if name == "view_reservation_options":
            if (
                self.tool_interface_version >= 21
                and env.config.reservation_contract_mode
                == "factorized_line_options_plan_set_v1"
            ):
                return ToolResult(False, {"error": f"unknown tool: {name}"})
            return ToolResult(
                True,
                env.view_reservation_options(
                    state,
                    request_id=str(args["request_id"]),
                    line_id=str(args["line_id"]),
                    view=str(args["view"]),
                    profile_family_id=(
                        None
                        if args.get("profile_family_id") is None
                        else str(args["profile_family_id"])
                    ),
                    promise_profile_id=(
                        None
                        if args.get("promise_profile_id") is None
                        else str(args["promise_profile_id"])
                    ),
                    cursor=int(args.get("cursor", 0)),
                    limit=int(args.get("limit", 20)),
                ),
            )
        if name == "quote_reservation_inquiries":
            return ToolResult(
                True,
                env.quote_reservation_inquiries(
                    state, self._object_array(args, "decisions")
                ),
            )
        if name == "confirm_reservation_plans":
            return ToolResult(
                True,
                env.confirm_reservation_plans(
                    state, self._object_array(args, "selections")
                ),
            )
        if name == "respond_reservation_service_offers":
            if self.tool_interface_version < 28:
                return ToolResult(False, {"error": f"unknown tool: {name}"})
            if set(args) != {"decisions"}:
                raise ValueError("only decisions is permitted")
            return ToolResult(True, env.respond_reservation_service_offers(state, self._object_array(args, "decisions")))
        if name == "stage_reservation_line":
            return ToolResult(
                True,
                env.stage_reservation_line(
                    state,
                    commitment_id=str(args["commitment_id"]),
                    line_id=str(args["line_id"]),
                    authorized_offer_id=str(args["authorized_offer_id"]),
                ),
            )
        if name == "respond_reservation_changes":
            return ToolResult(
                True,
                env.respond_reservation_changes(
                    state, self._object_array(args, "decisions")
                ),
            )
        if name == "request_reservation_contingency":
            if self.tool_interface_version < 33:
                return ToolResult(False, {"error": f"unknown tool: {name}"})
            if set(args) != {"inquiries"}:
                raise ValueError("only inquiries is permitted")
            return ToolResult(
                True,
                env.request_reservation_contingency(
                    state, self._object_array(args, "inquiries")
                ),
            )
        if name == "apply_reservation_contingency":
            if self.tool_interface_version < 33:
                return ToolResult(False, {"error": f"unknown tool: {name}"})
            if set(args) != {"decisions"}:
                raise ValueError("only decisions is permitted")
            return ToolResult(
                True,
                env.apply_reservation_contingency(
                    state, self._object_array(args, "decisions")
                ),
            )
        if name == "resolve_reservation_risks":
            return ToolResult(
                True,
                env.resolve_reservation_risks(
                    state, self._object_array(args, "actions")
                ),
            )
        if name == "cancel_reservation_commitment":
            if self.tool_interface_version < 25:
                return ToolResult(False, {"error": f"unknown tool: {name}"})
            unexpected = sorted(set(args) - {"commitment_id"})
            if unexpected:
                raise ValueError(f"unexpected arguments: {unexpected}")
            if "commitment_id" not in args:
                raise ValueError("commitment_id is required")
            return ToolResult(
                True,
                env.cancel_reservation_commitment(
                    state,
                    commitment_id=str(args["commitment_id"]),
                ),
            )
        if name == "view_rental_activity":
            horizon = int(args["horizon_days"])
            if not 1 <= horizon <= 30:
                raise ValueError("horizon_days must be in [1, 30]")
            end_day = args.get("end_day")
            if end_day is not None and int(end_day) < 1:
                raise ValueError("end_day must be positive")
            offer_ids = list(map(str, args.get("offer_ids", [])))
            unknown_offers = sorted(
                offer_id
                for offer_id in set(offer_ids)
                if offer_id not in env.offers
                or env.offers[offer_id].asset_kind != "durable"
            )
            if unknown_offers:
                return ToolResult(
                    False,
                    {"error": f"unknown durable offer_ids: {unknown_offers}"},
                )
            asset_ids = list(map(str, args.get("asset_ids", [])))
            known_asset_ids = {asset.asset_id for asset in state.assets}
            unknown_assets = sorted(set(asset_ids) - known_asset_ids)
            if unknown_assets:
                return ToolResult(
                    False,
                    {"error": f"unknown asset_ids: {unknown_assets}"},
                )
            event_types = list(map(str, args.get("event_types", [])))
            unknown_events = sorted(
                set(event_types) - (
                    set(RENTAL_ACTIVITY_EVENT_TYPES)
                    | ({"asset_disposed"} if env._v31_warehouse_capacity() else set())
                    | (
                        set(SUPPLIER_TRIAL_ACTIVITY_EVENT_TYPES)
                        if env._v37_supplier_trial()
                        else set()
                    )
                )
            )
            if unknown_events:
                return ToolResult(
                    False,
                    {
                        "error": f"unknown event_types: {unknown_events}",
                        "valid_event_types": list(RENTAL_ACTIVITY_EVENT_TYPES)
                        + (["asset_disposed"] if env._v31_warehouse_capacity() else [])
                        + (
                            list(SUPPLIER_TRIAL_ACTIVITY_EVENT_TYPES)
                            if env._v37_supplier_trial()
                            else []
                        ),
                    },
                )
            cursor = args.get("cursor")
            limit = int(args.get("limit", 25))
            if not 1 <= limit <= 50:
                raise ValueError("limit must be in [1, 50]")
            return ToolResult(
                True,
                env.view_rental_activity(
                    state,
                    horizon,
                    end_day=None if end_day is None else int(end_day),
                    product_types=[],
                    offer_ids=offer_ids,
                    asset_ids=asset_ids,
                    event_types=event_types,
                    cursor=None if cursor is None else str(cursor),
                    limit=limit,
                ),
            )
        if name == "view_liquidation_quotes":
            allowed = {"rental_asset_ids"}
            if env.config.retail_enabled:
                allowed.add("retail_items")
            unexpected = sorted(set(args) - allowed)
            if unexpected:
                raise ValueError(f"unexpected arguments: {unexpected}")
            return env.view_liquidation_quotes(
                state,
                self._string_array(args, "rental_asset_ids"),
                self._object_array(args, "retail_items", required=False),
            )
        if name == "sell_to_liquidation_dealer":
            allowed = {"rental_asset_ids"}
            if env.config.retail_enabled:
                allowed.add("retail_items")
            unexpected = sorted(set(args) - allowed)
            if unexpected:
                raise ValueError(f"unexpected arguments: {unexpected}")
            return env.sell_to_liquidation_dealer(
                state,
                self._string_array(args, "rental_asset_ids"),
                self._object_array(args, "retail_items", required=False),
            )
        if name == "set_prices":
            return env.set_prices(state, self._object_array(args, "prices"))
        if name == "set_rental_offer_status":
            return env.set_rental_offer_status(
                state, self._object_array(args, "items")
            )
        if name == "set_walk_in_capacity_limits":
            if self.tool_interface_version < 20:
                return ToolResult(False, {"error": f"unknown tool: {name}"})
            unexpected = sorted(set(args) - {"items"})
            if unexpected:
                raise ValueError(f"unexpected arguments: {unexpected}")
            return env.set_walk_in_capacity_limits(
                state, self._object_array(args, "items")
            )
        if name == "repair_or_retire_assets":
            return env.repair_or_retire(
                state,
                args.get("repair_asset_ids", []),
                args.get("retire_asset_ids", []),
            )
        if name == "schedule_maintenance":
            return env.schedule_maintenance(
                state, self._string_array(args, "asset_ids", required=True)
            )
        if name == "memory_list":
            unexpected = sorted(set(args) - {"limit", "offset", "prefix", "sort"})
            if unexpected:
                raise ValueError(f"unexpected arguments: {unexpected}")
            limit = int(args.get("limit", 20))
            offset = int(args.get("offset", 0))
            prefix = str(args.get("prefix", "")).strip().lower()
            sort = str(args.get("sort", "updated_desc"))
            if not 1 <= limit <= 50:
                raise ValueError("limit must be in [1, 50]")
            if offset < 0:
                raise ValueError("offset must be non-negative")
            if len(prefix) > 64:
                raise ValueError("prefix must contain at most 64 characters")
            if sort not in {"updated_desc", "name_asc"}:
                raise ValueError("sort must be updated_desc or name_asc")
            documents = [
                document
                for document in state.memory_documents.values()
                if not prefix or document.name.lower().startswith(prefix)
            ]
            if sort == "updated_desc":
                documents.sort(key=lambda row: (-row.updated_day, row.name))
            else:
                documents.sort(key=lambda row: row.name)
            page = documents[offset : offset + limit]
            next_offset = offset + len(page)
            return ToolResult(
                True,
                {
                    "documents": [
                        {
                            "name": document.name,
                            "description": document.description,
                            "characters": len(document.content),
                            "updated_day": document.updated_day,
                        }
                        for document in page
                    ],
                    "total_documents": len(state.memory_documents),
                    "matched_documents": len(documents),
                    "returned_documents": len(page),
                    "offset": offset,
                    "has_more": next_offset < len(documents),
                    "next_offset": (
                        next_offset if next_offset < len(documents) else None
                    ),
                },
            )
        if name == "memory_search":
            unexpected = sorted(set(args) - {"query", "limit", "offset"})
            if unexpected:
                raise ValueError(f"unexpected arguments: {unexpected}")
            query = str(args["query"]).strip()
            limit = int(args.get("limit", 20))
            offset = int(args.get("offset", 0))
            if not query:
                raise ValueError("query must not be empty")
            if len(query) > 200:
                raise ValueError("query must contain at most 200 characters")
            if not 1 <= limit <= 50:
                raise ValueError("limit must be in [1, 50]")
            if offset < 0:
                raise ValueError("offset must be non-negative")
            normalized_query = unicodedata.normalize("NFKC", query).casefold()
            terms = normalized_query.split()
            matches: list[tuple[tuple[int, int, int, int, int, str], MemoryDocument, list[str]]] = []
            for document in state.memory_documents.values():
                fields = {
                    "name": unicodedata.normalize("NFKC", document.name).casefold(),
                    "description": unicodedata.normalize(
                        "NFKC", document.description
                    ).casefold(),
                    "content": unicodedata.normalize(
                        "NFKC", document.content
                    ).casefold(),
                }
                if not all(any(term in value for value in fields.values()) for term in terms):
                    continue
                hits = {
                    field: sum(term in value for term in terms)
                    for field, value in fields.items()
                }
                matched_fields = [field for field in fields if hits[field] > 0]
                rank = (
                    -int(fields["name"] == normalized_query),
                    -hits["name"],
                    -hits["description"],
                    -hits["content"],
                    -document.updated_day,
                    document.name,
                )
                matches.append((rank, document, matched_fields))
            matches.sort(key=lambda row: row[0])
            page = matches[offset : offset + limit]
            next_offset = offset + len(page)
            return ToolResult(
                True,
                {
                    "query": query,
                    "documents": [
                        {
                            "name": document.name,
                            "description": document.description,
                            "characters": len(document.content),
                            "updated_day": document.updated_day,
                            "matched_fields": matched_fields,
                        }
                        for _, document, matched_fields in page
                    ],
                    "total_documents": len(state.memory_documents),
                    "matched_documents": len(matches),
                    "returned_documents": len(page),
                    "offset": offset,
                    "has_more": next_offset < len(matches),
                    "next_offset": next_offset if next_offset < len(matches) else None,
                },
            )
        if name in {"memory_read", "memory_write", "memory_delete"}:
            raw_name = str(args["name"])
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", raw_name):
                return ToolResult(False, {"error": "memory name must use letters, digits, underscore, or dash"})
            key = raw_name.lower()
            if name == "memory_read":
                document = state.memory_documents.get(key)
                if document is None:
                    return ToolResult(False, {"error": "memory document not found", "name": key})
                return ToolResult(
                    True,
                    {
                        "name": document.name,
                        "description": document.description,
                        "content": document.content,
                        "created_day": document.created_day,
                        "updated_day": document.updated_day,
                    },
                )
            if name == "memory_delete":
                document = state.memory_documents.pop(key, None)
                if document is None:
                    return ToolResult(False, {"error": "memory document not found", "name": key})
                return ToolResult(True, {"name": key, "deleted": True})
            description = str(args["description"]).strip()
            if not 1 <= len(description) <= 200:
                return ToolResult(False, {"error": "memory description must contain 1-200 characters"})
            content = str(args["content"])
            if len(content) > 40000:
                return ToolResult(False, {"error": "memory content exceeds 40,000 characters"})
            existing = state.memory_documents.get(key)
            state.memory_documents[key] = MemoryDocument(
                name=key,
                description=description,
                content=content,
                created_day=state.day if existing is None else existing.created_day,
                updated_day=state.day,
            )
            return ToolResult(
                True,
                {
                    "name": key,
                    "description": description,
                    "characters": len(content),
                    "action": "created" if existing is None else "updated",
                },
            )
        if name == "close_business_day":
            unexpected = sorted(set(args) - {"handoff_note"})
            if unexpected:
                return ToolResult(
                    False,
                    {"error": f"unexpected arguments: {unexpected}"},
                )
            raw_note = args.get("handoff_note")
            if raw_note is not None and not isinstance(raw_note, str):
                return ToolResult(
                    False, {"error": "handoff_note must be a string"}
                )
            note = None if raw_note is None else raw_note.strip()
            if note and len(note) > 3000:
                return ToolResult(
                    False,
                    {"error": "handoff_note exceeds 3,000 characters"},
                )
            return env.close_business_day(state, handoff_note=note or None)
        return ToolResult(False, {"error": f"unknown tool: {name}"})
