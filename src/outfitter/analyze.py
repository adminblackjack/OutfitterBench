"""Materialize operating tables and plots for an OutfitterBench run."""

from __future__ import annotations

import copy

from .analysis_plots import PLOT_FILES, render_plots as _render_plots

import csv
import json
import sqlite3
import statistics
import hashlib
import shutil
import tempfile
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .config import OutfitterConfig
from .data import OutfitterDataset
from .environment import RENTAL_HOLDING_RATE, OutfitterEnvironment
from .models import WorldState
from .reservation_feedback import (
    EQUIPMENT_CONFIGURATION_TAG,
    EQUIPMENT_FIT_CONCERN,
)
from .runtime.common import (
    RESERVATION_REUSE_CATEGORY_ADJUSTMENT,
    RESERVATION_STORAGE_CATEGORY_ADJUSTMENT,
)


REPORT_FORMAT = "outfitter-analysis-v6-executed-call-identity"
RESERVATION_FORBIDDEN_PUBLIC_TERMS = (
    "quote_context_by_date",
    "service_offer_audit",
    "market_reference_line_day",
    "quote_context_by_line",
    "candidate_deposit_by_line",
    "latent_price_ceiling",
    "latent_offer_ids",
    "reference_daily_price",
    "quote_at_reference_rates",
    "quote_at_current_posted_rates",
    "line_overrides",
    "inquiry_latent_offer_ids_by_line",
    "inquiry_latent_price_ceiling_by_line",
    "authorized_price_ceiling_by_line",
    "recovery_offer_ids_by_line",
    "recovery_price_ceiling_by_line",
    "inquiry_public_envelope_by_line",
    "inquiry_profile_member_ids_by_line",
    "inquiry_profile_price_ceiling_by_line",
    "inquiry_profile_deposit_by_line",
    "latent_snapshot_provenance",
    "customer_segment",
    "shared_price_beta",
    "storage_burden_base",
    "future_use_expectation_base",
    "return_burden_multiplier",
    "request_budget_multiplier",
    "price_curve_gains",
    "price_action_gain",
    "equipment_action_gain",
    "equipment_action_gain_by_line",
    "price_tag_probability",
    "equipment_tag_probability",
    "price_tag_uniform",
    "equipment_tag_uniform",
)
COST_FIELDS = (
    "purchase_cost",
    "supplier_trial_handling_fee",
    "supplier_trial_usage_fee",
    "cleaning_cost",
    "repair_cost",
    "maintenance_cost",
    "retail_holding_cost",
    "rental_holding_cost",
    "rental_idle_holding_cost",
    "reservation_compensation",
    "operating_fee",
    "model_cost",
)


def _number(values: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Read a numeric persisted field without making old run snapshots fail."""
    value = values.get(key, default)
    return float(default if value is None else value)


def _daily_retail_cogs(report: dict[str, Any]) -> float:
    """Recover retail COGS from the per-service fulfillment breakdown.

    Retail COGS was intentionally not part of the original compact financial
    ledger.  The detailed fulfillment breakdown is present in those runs, so
    prefer it while accepting a future explicit financial field.
    """
    financial = report.get("financial", {})
    for field in ("retail_cogs", "retail_cost_of_goods_sold"):
        if field in financial:
            return _number(financial, field)
    return round(
        sum(
            _number(values, "retail_cogs")
            for metric in report.get("service_metrics", {}).values()
            for values in metric.get("product_fulfillment_breakdown", {}).values()
        ),
        2,
    )


def _rental_depreciation_by_day(state: WorldState) -> dict[int, float]:
    """Derive rental-use depreciation for legacy runs from recorded contracts.

    New environments may persist an exact daily amount in ``financial``.  In
    older snapshots, assets accrue their full used-life increment when a
    contract starts, so assigning ``unit_cost / service_life`` times duration
    to that start day matches the state transition and avoids treating a fleet
    purchase as an operating expense.
    """
    assets = {asset.asset_id: asset for asset in state.assets}
    values: defaultdict[int, float] = defaultdict(float)
    for contract in state.contracts:
        asset = assets.get(contract.asset_id)
        if asset is None or not asset.service_life_limit_days:
            continue
        if asset.ownership_type != "owned":
            continue
        wear = int(contract.duration_days) / float(asset.service_life_limit_days)
        if asset.service_life_limit_cycles is not None:
            wear += 1.0 / float(asset.service_life_limit_cycles)
        values[int(contract.start_day)] += float(asset.unit_cost) * wear
    return {day: round(value, 2) for day, value in values.items()}


def _read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in values:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in values:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _classify_repeated_read_only_calls(
    rows: list[dict[str, Any]],
    models: list[dict[str, Any]],
    *,
    signature: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Classify a re-read using the exact context sent before that call.

    A repeated result is wasteful only when an earlier equivalent tool result
    is still present in the model request. A same-response parallel duplicate
    is separate because neither result was available when the calls were
    generated. If no earlier equivalent call ID survives in the request, the
    call is a context-eviction refetch rather than an in-context repetition.
    """

    model_by_call_id: dict[str, dict[str, Any]] = {}
    for model in models:
        for call_id in model.get("tool_call_ids", []):
            if call_id:
                model_by_call_id[str(call_id)] = model
    prior_by_signature: defaultdict[Any, list[str]] = defaultdict(list)
    by_call_id: dict[str, str] = {}
    counts: Counter[str] = Counter()
    characters: Counter[str] = Counter()
    for row in rows:
        key = signature(row)
        if key is None:
            continue
        call_id = str(row.get("call_id") or "")
        prior_call_ids = prior_by_signature[key]
        if prior_call_ids:
            model = model_by_call_id.get(call_id)
            if model is None:
                category = "unattributed"
            else:
                current_model_id = int(model["id"])
                prior_same_model = any(
                    int(model_by_call_id[prior]["id"]) == current_model_id
                    for prior in prior_call_ids
                    if prior in model_by_call_id
                )
                retained = set(model.get("context_tool_call_ids", []))
                if prior_same_model:
                    category = "same_model_batch_repetition"
                elif any(prior in retained for prior in prior_call_ids):
                    category = "repeated_while_still_in_context"
                else:
                    category = "refetched_after_context_eviction"
            by_call_id[call_id] = category
            counts[category] += 1
            characters[category] += int(row.get("result_characters", 0))
        prior_call_ids.append(call_id)
    categories = (
        "repeated_while_still_in_context",
        "refetched_after_context_eviction",
        "same_model_batch_repetition",
        "unattributed",
    )
    return (
        {
            "same_signature_reoccurrences": sum(counts.values()),
            **{
                category: {
                    "records": int(counts[category]),
                    "characters": int(characters[category]),
                }
                for category in categories
            },
        },
        by_call_id,
    )



def _load_run(
    run_dir: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], OutfitterConfig, OutfitterDataset, OutfitterEnvironment, WorldState]:
    path = Path(run_dir).expanduser().resolve()
    manifest = _read_json(path / "manifest.json", None)
    if not isinstance(manifest, dict) or manifest.get("format") != "outfitter-run-v1":
        raise ValueError(f"not a supported OutfitterBench run: {path}")
    values = dict(manifest["config"])
    values["data_dir"] = str(path / manifest.get("runtime_data", "data"))
    config = OutfitterConfig.from_dict(values)
    dataset = OutfitterDataset.load(config.data_dir)
    environment = OutfitterEnvironment(config, dataset)
    with sqlite3.connect(path / "run.sqlite3", timeout=30) as connection:
        raw = connection.execute("SELECT value FROM state WHERE id=1").fetchone()
    if raw is None:
        raise RuntimeError("run state is missing")
    state = WorldState.from_dict(json.loads(raw[0]))
    return path, manifest, _read_json(path / "summary.json", {}), config, dataset, environment, state


def _load_events(
    path: Path, *, tool_call_protocol: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    models: list[dict[str, Any]] = []
    tools: list[dict[str, Any]] = []
    with sqlite3.connect(path / "run.sqlite3", timeout=30) as connection:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        from .call_identity import load_identity, response_message, response_finish_reason
        identity = load_identity(connection, tool_call_protocol)
        for row in connection.execute(
            "SELECT id,day,input_tokens,output_tokens,request,response "
            "FROM model_calls ORDER BY id"
        ):
            raw_request = json.loads(row[4])
            raw_response = json.loads(row[5])
            request_messages = list(raw_request.get("messages") or [])
            omitted_messages = 0
            truncation_notice = False
            for message in request_messages:
                content = str(message.get("content") or "")
                suffix = " older messages from today no longer fit"
                if suffix in content:
                    try:
                        omitted_messages = max(
                            omitted_messages,
                            int(content.split(suffix, 1)[0]),
                        )
                    except ValueError:
                        pass
                truncation_notice = truncation_notice or (
                    "hit its output limit" in content
                )
            raw_message = response_message(raw_response)
            raw_calls = list(raw_message.get("tool_calls") or [])
            raw_tool_calls = len(raw_calls)
            finish_reason = response_finish_reason(raw_response)
            malformed_call_ids: list[str] = []
            if str(finish_reason) == "length":
                for call in raw_calls:
                    function = call.get("function") or {}
                    arguments = function.get("arguments", "{}")
                    try:
                        decoded = (
                            json.loads(arguments)
                            if isinstance(arguments, str)
                            else arguments
                        )
                    except json.JSONDecodeError:
                        decoded = None
                    if not isinstance(decoded, dict):
                        malformed_call_ids.append(str(call.get("id", "")))
            models.append(
                {
                    "id": int(row[0]),
                    "day": int(row[1]),
                    "input_tokens": int(row[2]),
                    "output_tokens": int(row[3]),
                    "context_message_characters": len(
                        json.dumps(request_messages, ensure_ascii=False)
                    ),
                    "sliding_window_omitted_messages": omitted_messages,
                    "generation_truncation_notice_present": truncation_notice,
                    "raw_tool_calls": raw_tool_calls,
                    "finish_reason": (
                        "missing" if finish_reason is None else str(finish_reason)
                    ),
                    "generation_truncated": str(finish_reason) == "length",
                    "generation_truncated_tool_call_ids": malformed_call_ids,
                    "raw_tool_call_ids": [str(call.get("id", "")) for call in raw_calls],
                    "tool_call_ids": identity['model_ids'].get(int(row[0]), []),
                    "identity_status": identity['statuses'].get(int(row[0]), 'unknown'),
                    "context_tool_call_ids": [
                        str(message.get("tool_call_id"))
                        for message in request_messages
                        if message.get("role") == "tool"
                        and message.get("tool_call_id") is not None
                    ],
                    "discarded_tool_calls": (
                        max(0, raw_tool_calls - 1)
                        if tool_call_protocol == "first_only"
                        else 0
                    ),
                    "read_only_tool_batches": 0,
                    "batched_read_only_tool_calls": 0,
                    "rejected_tool_batches": 0,
                    "rejected_batched_tool_calls": 0,
                }
            )
        for row in connection.execute(
            "SELECT call_id,day,name,arguments,result FROM tool_calls ORDER BY rowid"
        ):
            result = json.loads(row[4])
            tools.append(
                {
                    "call_id": str(row[0]),
                    "day": int(row[1]),
                    "name": str(row[2]),
                    "arguments": json.loads(row[3]),
                    "ok": bool(result.get("ok")),
                    "data": result.get("data", {}),
                    "result_characters": len(row[4]),
                }
            )
    if tool_call_protocol == "read_only_batch_v1":
        tools_by_id = {row["call_id"]: row for row in tools}
        for model in models:
            call_ids = model["tool_call_ids"]
            if len(call_ids) <= 1:
                continue
            # A length-truncated response may retain raw call fragments in the
            # provider response while the controller deliberately executes
            # none of them. Count only fully persisted batches; raw generation
            # attempts are reported separately.
            if not all(call_id in tools_by_id for call_id in call_ids):
                continue
            rejected = any(
                tools_by_id.get(call_id, {}).get("data", {}).get("error")
                == "tool_batch_policy_violation"
                for call_id in call_ids
            )
            if rejected:
                model["rejected_tool_batches"] = 1
                model["rejected_batched_tool_calls"] = len(call_ids)
            else:
                model["read_only_tool_batches"] = 1
                model["batched_read_only_tool_calls"] = len(call_ids)
    agent_call_ids = identity['agent_ids']
    generation_truncated_call_ids = {
        call_id
        for model in models
        for call_id in model["generation_truncated_tool_call_ids"]
        if call_id
    }
    tools_by_id = {row["call_id"]: row for row in tools}
    for row in tools:
        row["origin"] = (
            "agent" if row["call_id"] in agent_call_ids else
            "automatic" if row["call_id"] in identity['automatic_ids'] else "unknown"
        )
        row["generation_truncation"] = (
            row["call_id"] in generation_truncated_call_ids
        )
    for model in models:
        executed = [
            tools_by_id[call_id]
            for call_id in model["tool_call_ids"]
            if call_id in tools_by_id
        ]
        model["invalid_action_episode"] = int(
            any(not row["ok"] for row in executed)
            and not model["generation_truncated_tool_call_ids"]
        )
    return models, tools, integrity


def _reservation_protocol_audit(
    path: Path,
    state: WorldState,
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Cross-check the persisted public protocol, not just final counters.

    Final state alone cannot distinguish an explicit complete quote from a
    legacy shortcut, nor can it detect an incorrect opening inbox summary.
    Keep this audit additive so old runs remain analyzable.
    """

    run_manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    implicit_optional_omit = int(
        run_manifest.get("controller", {}).get("tool_interface_version", 0)
    ) >= 32

    message_values: list[str] = []
    model_requests: list[str] = []
    model_responses: list[str] = []
    with sqlite3.connect(path / "run.sqlite3", timeout=30) as connection:
        message_values = [
            str(row[0])
            for row in connection.execute("SELECT value FROM messages ORDER BY seq")
        ]
        for request, response in connection.execute(
            "SELECT request,response FROM model_calls ORDER BY id"
        ):
            model_requests.append(str(request))
            model_responses.append(str(response))

    expected_opening_counts: Counter[int] = Counter(
        int(request.created_day) for request in state.reservation_requests if not request.service_offer_terms
    )
    v32_opening = any(r.pickup_date_options for r in state.reservation_requests)
    expected_service_counts = Counter(int(r.created_day) for r in state.reservation_requests if r.service_offer_terms)
    opening_rows: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    for raw in message_values:
        try:
            message = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, str):
            continue
        marker = content.find("opening business report")
        if marker < 0:
            continue
        start = content.find("{", marker)
        if start < 0:
            continue
        try:
            opening = decoder.raw_decode(content[start:])[0]
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(opening, dict) or "reservation_inbox_count" not in opening:
            continue
        day = int(opening.get("day", len(opening_rows) + 1))
        reported = int(opening["reservation_inbox_count"])
        expected = int(expected_opening_counts[day])
        opening_rows.append(
            {
                "day": day,
                "reported_pending_quote_count": reported,
                "expected_created_inquiry_count": expected,
                **({"reported_pending_service_offer_count": int(opening.get("reservation_service_offer_count", 0)),
                    "expected_created_service_offer_count": expected_service_counts[day]} if v32_opening else {}),
                "matches": reported == expected and (not v32_opening or int(opening.get("reservation_service_offer_count", 0)) == expected_service_counts[day]),
            }
        )

    requests_by_id = {
        request.request_id: request for request in state.reservation_requests
    }
    action_counts: Counter[str] = Counter()
    completeness_violations: list[dict[str, Any]] = []
    explicit_quotes = 0
    complete_explicit_quotes = 0
    failed_quote_tool_calls = 0
    successful_quote_tool_calls = 0
    for call in tools:
        if call["name"] != "quote_reservation_inquiries":
            continue
        if not call["ok"]:
            failed_quote_tool_calls += 1
            continue
        successful_quote_tool_calls += 1
        decisions = call["arguments"].get("decisions", [])
        for index, decision in enumerate(decisions):
            action = str(decision.get("action", "missing"))
            if action == "quote":
                action_counts["explicit_quote"] += 1
                explicit_quotes += 1
                request_id = str(decision.get("request_id", ""))
                request = requests_by_id.get(request_id)
                expected_ids = (
                    {line.line_id for line in request.lines}
                    if request is not None
                    else set()
                )
                profile_quote = (
                    request is not None
                    and request.reservation_contract_mode
                    in {
                        "promise_profiles_v1",
                        "adaptive_pairwise_exact_quotes_v1",
                        "complete_contract_options_v1",
                        "factorized_line_options_plan_set_v1",
                    }
                )
                pairwise_quote = (
                    request is not None
                    and request.reservation_contract_mode
                    == "adaptive_pairwise_exact_quotes_v1"
                )
                complete_contract_quote = (
                    request is not None
                    and request.reservation_contract_mode
                    == "complete_contract_options_v1"
                )
                factorized_plan_quote = (
                    request is not None
                    and request.reservation_contract_mode
                    == "factorized_line_options_plan_set_v1"
                )
                if complete_contract_quote:
                    problems: list[str] = []
                    raw_contracts = decision.get("contract_options")
                    submitted_ids: list[str] = []
                    if not isinstance(raw_contracts, list) or not 1 <= len(
                        raw_contracts
                    ) <= 2:
                        problems.append("invalid_complete_contract_option_count")
                    else:
                        for contract_index, contract in enumerate(raw_contracts):
                            raw_lines = (
                                contract.get("lines")
                                if isinstance(contract, dict)
                                else None
                            )
                            if not isinstance(raw_lines, list):
                                problems.append(
                                    f"contract_lines_not_a_list:{contract_index}"
                                )
                                continue
                            line_ids = [
                                str(row.get("line_id", ""))
                                for row in raw_lines
                                if isinstance(row, dict)
                            ]
                            submitted_ids.extend(line_ids)
                            if len(line_ids) != len(raw_lines):
                                problems.append(
                                    f"non_object_contract_line:{contract_index}"
                                )
                            if len(line_ids) != len(set(line_ids)):
                                problems.append(
                                    f"duplicate_line_id:{contract_index}"
                                )
                            if set(line_ids) != expected_ids:
                                problems.append(
                                    f"line_id_set_mismatch:{contract_index}"
                                )
                    if request is None:
                        problems.append("unknown_request_id")
                    if problems:
                        completeness_violations.append(
                            {
                                "call_id": call["call_id"],
                                "decision_index": index,
                                "request_id": request_id,
                                "problems": problems,
                                "expected_line_ids": sorted(expected_ids),
                                "submitted_line_ids": submitted_ids,
                            }
                        )
                    else:
                        complete_explicit_quotes += 1
                    continue
                quote_rows = decision.get(
                    "lines" if profile_quote else "line_prices"
                )
                submitted_ids = (
                    [
                        str(row.get("line_id", ""))
                        for row in quote_rows
                        if isinstance(row, dict)
                    ]
                    if isinstance(quote_rows, list)
                    else []
                )
                problems: list[str] = []
                if request is None:
                    problems.append("unknown_request_id")
                if not isinstance(quote_rows, list):
                    problems.append(
                        "profile_lines_not_a_list"
                        if profile_quote
                        else "line_prices_not_a_list"
                    )
                if len(submitted_ids) != len(set(submitted_ids)):
                    problems.append("duplicate_line_id")
                if set(submitted_ids) != expected_ids:
                    problems.append("line_id_set_mismatch")
                if profile_quote and isinstance(quote_rows, list):
                    lines_by_id = {
                        line.line_id: line
                        for line in request.lines
                    }
                    for row in quote_rows:
                        if not isinstance(row, dict):
                            continue
                        line_id = str(row.get("line_id", ""))
                        if factorized_plan_quote:
                            options = row.get("options")
                            if not isinstance(options, list):
                                problems.append(
                                    f"factorized_options_not_a_list:{line_id}"
                                )
                                continue
                            quote_options = [
                                option
                                for option in options
                                if isinstance(option, dict)
                                and option.get("disposition") == "quote"
                            ]
                            omit_options = [
                                option
                                for option in options
                                if isinstance(option, dict)
                                and option.get("disposition") == "omit"
                            ]
                            line = lines_by_id.get(line_id)
                            if line is not None and line.required:
                                if not 1 <= len(quote_options) <= 2 or omit_options:
                                    problems.append(
                                        f"invalid_required_factorized_options:{line_id}"
                                    )
                            elif line is not None and (
                                len(quote_options) > 2
                                or len(omit_options)
                                not in (
                                    {0, 1}
                                    if implicit_optional_omit
                                    else {1}
                                )
                            ):
                                problems.append(
                                    f"invalid_optional_factorized_options:{line_id}"
                                )
                            exact_profiles = [
                                (
                                    str(option.get("offer_id", "")),
                                    str(option.get("promise_profile_id", "")),
                                )
                                for option in quote_options
                            ]
                            if len(exact_profiles) != len(set(exact_profiles)):
                                problems.append(
                                    f"duplicate_factorized_exact_profile:{line_id}"
                                )
                            for option in quote_options:
                                profile_id = str(
                                    option.get("promise_profile_id", "")
                                )
                                if (
                                    line is not None
                                    and profile_id not in line.eligible_profile_ids
                                ):
                                    problems.append(
                                        f"ineligible_profile:{line_id}"
                                    )
                        elif pairwise_quote:
                            options = row.get("quote_options")
                            if not isinstance(options, list) or not 1 <= len(
                                options
                            ) <= 2:
                                problems.append(
                                    f"invalid_pairwise_option_count:{line_id}"
                                )
                                continue
                            exact_ids = [
                                str(option.get("offer_id", ""))
                                for option in options
                                if isinstance(option, dict)
                            ]
                            if len(exact_ids) != len(options):
                                problems.append(
                                    f"non_object_pairwise_option:{line_id}"
                                )
                            if len(exact_ids) != len(set(exact_ids)):
                                problems.append(
                                    f"duplicate_pairwise_exact:{line_id}"
                                )
                            for option in options:
                                if not isinstance(option, dict):
                                    continue
                                profile_id = str(
                                    option.get("promise_profile_id", "")
                                )
                                if (
                                    line_id in lines_by_id
                                    and profile_id
                                    not in lines_by_id[
                                        line_id
                                    ].eligible_profile_ids
                                ):
                                    problems.append(
                                        f"ineligible_profile:{line_id}"
                                    )
                        else:
                            profile_id = str(
                                row.get("promise_profile_id", "")
                            )
                            if (
                                line_id in lines_by_id
                                and profile_id
                                not in lines_by_id[line_id].eligible_profile_ids
                            ):
                                problems.append(
                                    f"ineligible_profile:{line_id}"
                                )
                if problems:
                    completeness_violations.append(
                        {
                            "call_id": call["call_id"],
                            "decision_index": index,
                            "request_id": request_id,
                            "problems": problems,
                            "expected_line_ids": sorted(expected_ids),
                            "submitted_line_ids": submitted_ids,
                        }
                    )
                else:
                    complete_explicit_quotes += 1
            elif action == "decline":
                action_counts["decline"] += 1
                if (
                    "line_prices" in decision
                    or "lines" in decision
                    or "contract_options" in decision
                ):
                    completeness_violations.append(
                        {
                            "call_id": call["call_id"],
                            "decision_index": index,
                            "request_id": str(decision.get("request_id", "")),
                            "problems": ["decline_carried_quote_lines"],
                        }
                    )
            elif action in {
                "quote_at_reference_rates",
                "quote_at_current_posted_rates",
            }:
                action_counts["legacy_shortcut_quote"] += 1
            else:
                action_counts[f"unknown:{action}"] += 1

    tool_surface = [
        json.dumps(
            {"arguments": call["arguments"], "result": call["data"]},
            ensure_ascii=False,
            sort_keys=True,
        )
        for call in tools
    ]
    surfaces = {
        "messages": message_values,
        "tool_calls": tool_surface,
        "model_requests": model_requests,
        "model_responses": model_responses,
        "system_prompt": [
            (path / "system_prompt.md").read_text(encoding="utf-8")
            if (path / "system_prompt.md").is_file()
            else ""
        ],
    }
    forbidden_hits: dict[str, dict[str, int]] = {}
    for term in RESERVATION_FORBIDDEN_PUBLIC_TERMS:
        by_surface = {
            surface: sum(text.count(term) for text in values)
            for surface, values in surfaces.items()
        }
        forbidden_hits[term] = {
            **by_surface,
            "total": sum(by_surface.values()),
        }

    shared_mismatches: Counter[str] = Counter()
    storage_adjustment_violations = 0
    reuse_adjustment_violations = 0
    snapshot_violations: list[dict[str, Any]] = []
    snapshot_provenance = Counter(
        request.latent_snapshot_provenance
        for request in state.reservation_requests
    )
    for request in state.reservation_requests:
        contexts = list(request.quote_context_by_line.values())
        shared_expectations = {
            "customer_segment": request.customer_segment,
            "price_beta": request.shared_price_beta,
            "forecast_trust": request.forecast_trust,
            "return_burden_multiplier": request.return_burden_multiplier,
        }
        if (
            request.latent_snapshot_provenance
            == "immutable_profile_inquiry_no_ceiling_v2"
        ):
            shared_expectations["request_budget_multiplier"] = (
                request.request_budget_multiplier
            )
        for field, expected in shared_expectations.items():
            if any(context.get(field) != expected for context in contexts):
                shared_mismatches[field] += 1
        for context in contexts:
            service_key = str(context.get("service_key", ""))
            expected_storage = round(
                min(
                    1.0,
                    max(
                        0.0,
                        request.storage_burden_base
                        + RESERVATION_STORAGE_CATEGORY_ADJUSTMENT.get(
                            service_key, 0.0
                        ),
                    ),
                ),
                6,
            )
            expected_reuse = round(
                min(
                    1.0,
                    max(
                        0.0,
                        request.future_use_expectation_base
                        + RESERVATION_REUSE_CATEGORY_ADJUSTMENT.get(
                            service_key, 0.0
                        ),
                    ),
                ),
                6,
            )
            if abs(float(context.get("storage_burden", -1.0)) - expected_storage) > 1e-9:
                storage_adjustment_violations += 1
            if abs(float(context.get("future_use_expectation", -1.0)) - expected_reuse) > 1e-9:
                reuse_adjustment_violations += 1

        if request.latent_snapshot_provenance in {
            "immutable_profile_inquiry_v1",
            "immutable_profile_inquiry_no_ceiling_v2",
        }:
            ceilingless_profile = (
                request.latent_snapshot_provenance
                == "immutable_profile_inquiry_no_ceiling_v2"
            )
            line_ids = {line.line_id for line in request.lines}
            problems: list[str] = []
            for name, mapping in (
                (
                    "profile_public_envelope",
                    request.inquiry_public_envelope_by_line,
                ),
                (
                    "profile_members",
                    request.inquiry_profile_member_ids_by_line,
                ),
                (
                    "profile_ceilings",
                    request.inquiry_profile_price_ceiling_by_line,
                ),
                (
                    "profile_deposits",
                    request.inquiry_profile_deposit_by_line,
                ),
            ):
                if set(mapping) != line_ids:
                    problems.append(f"{name}_line_keys_mismatch")
            for line in request.lines:
                eligible = set(line.eligible_profile_ids)
                members_by_profile = request.inquiry_profile_member_ids_by_line.get(
                    line.line_id, {}
                )
                ceilings_by_profile = request.inquiry_profile_price_ceiling_by_line.get(
                    line.line_id, {}
                )
                deposits_by_profile = request.inquiry_profile_deposit_by_line.get(
                    line.line_id, {}
                )
                if not eligible or set(members_by_profile) != eligible:
                    problems.append(f"profile_member_keys:{line.line_id}")
                if set(ceilings_by_profile) != eligible:
                    problems.append(f"profile_ceiling_keys:{line.line_id}")
                if set(deposits_by_profile) != eligible:
                    problems.append(f"profile_deposit_keys:{line.line_id}")
                if line.promise_profile_id is None:
                    continue
                selected_members = set(
                    members_by_profile.get(line.promise_profile_id, [])
                )
                selected_ceilings = set(
                    ceilings_by_profile.get(line.promise_profile_id, {})
                )
                authorized = set(
                    request.authorized_offer_ids_by_line.get(line.line_id, [])
                )
                recovery = set(
                    request.recovery_offer_ids_by_line.get(line.line_id, [])
                )
                if not authorized <= selected_members:
                    problems.append(f"profile_authorized_not_member:{line.line_id}")
                if authorized & recovery:
                    problems.append(f"profile_authorized_recovery_overlap:{line.line_id}")
                if ceilingless_profile:
                    if selected_ceilings:
                        problems.append(
                            f"ceilingless_profile_has_inquiry_ceiling:{line.line_id}"
                        )
                    if not authorized | recovery <= selected_members:
                        problems.append(
                            f"profile_post_quote_not_member:{line.line_id}"
                        )
                    if request.authorized_price_ceiling_by_line.get(
                        line.line_id, {}
                    ) or request.recovery_price_ceiling_by_line.get(
                        line.line_id, {}
                    ):
                        problems.append(
                            f"ceilingless_profile_has_post_quote_ceiling:{line.line_id}"
                        )
                elif not authorized | recovery <= selected_ceilings:
                    problems.append(f"profile_post_quote_not_frozen:{line.line_id}")
            if problems:
                snapshot_violations.append(
                    {"request_id": request.request_id, "problems": problems}
                )
            continue
        if request.latent_snapshot_provenance != "immutable_inquiry_v1":
            continue
        line_ids = {line.line_id for line in request.lines}
        problems: list[str] = []
        if set(request.inquiry_latent_offer_ids_by_line) != line_ids:
            problems.append("inquiry_offer_line_keys_mismatch")
        if set(request.inquiry_latent_price_ceiling_by_line) != line_ids:
            problems.append("inquiry_ceiling_line_keys_mismatch")
        if request.commitment_id:
            if set(request.authorized_offer_ids_by_line) != line_ids:
                problems.append("authorized_line_keys_mismatch")
            if set(request.authorized_price_ceiling_by_line) != line_ids:
                problems.append("authorized_ceiling_line_keys_mismatch")
            if set(request.recovery_offer_ids_by_line) != line_ids:
                problems.append("recovery_line_keys_mismatch")
            if set(request.recovery_price_ceiling_by_line) != line_ids:
                problems.append("recovery_ceiling_line_keys_mismatch")
        elif any(
            (
                request.authorized_offer_ids_by_line,
                request.authorized_price_ceiling_by_line,
                request.recovery_offer_ids_by_line,
                request.recovery_price_ceiling_by_line,
            )
        ):
            problems.append("unconfirmed_request_has_post_quote_sets")
        for line_id in line_ids:
            inquiry = set(
                request.inquiry_latent_offer_ids_by_line.get(line_id, [])
            )
            inquiry_ceiling_ids = set(
                request.inquiry_latent_price_ceiling_by_line.get(line_id, {})
            )
            authorized = set(
                request.authorized_offer_ids_by_line.get(line_id, [])
            )
            authorized_ceiling_ids = set(
                request.authorized_price_ceiling_by_line.get(line_id, {})
            )
            recovery = set(request.recovery_offer_ids_by_line.get(line_id, []))
            recovery_ceiling_ids = set(
                request.recovery_price_ceiling_by_line.get(line_id, {})
            )
            if inquiry != inquiry_ceiling_ids:
                problems.append(f"inquiry_offer_ceiling_keys:{line_id}")
            if authorized & recovery:
                problems.append(f"authorized_recovery_overlap:{line_id}")
            if not authorized | recovery <= inquiry:
                problems.append(f"post_quote_not_in_inquiry:{line_id}")
            if authorized != authorized_ceiling_ids:
                problems.append(f"authorized_offer_ceiling_keys:{line_id}")
            if recovery != recovery_ceiling_ids:
                problems.append(f"recovery_offer_ceiling_keys:{line_id}")
        if problems:
            snapshot_violations.append(
                {"request_id": request.request_id, "problems": problems}
            )

    return {
        "opening_reports_audited": len(opening_rows),
        "opening_inbox_mismatch_count": sum(
            not row["matches"] for row in opening_rows
        ),
        "opening_inbox_rows": opening_rows,
        "successful_quote_tool_calls": successful_quote_tool_calls,
        "failed_quote_tool_calls": failed_quote_tool_calls,
        "successful_decision_counts": dict(sorted(action_counts.items())),
        "explicit_quotes_audited": explicit_quotes,
        "complete_explicit_quotes": complete_explicit_quotes,
        "explicit_quote_completeness_violations": completeness_violations,
        "forbidden_public_term_hits": forbidden_hits,
        "forbidden_public_term_hit_total": sum(
            values["total"] for values in forbidden_hits.values()
        ),
        "bundle_requests_audited": len(state.reservation_requests),
        "bundle_shared_field_mismatch_counts": dict(
            sorted(shared_mismatches.items())
        ),
        "storage_category_adjustment_violation_count": (
            storage_adjustment_violations
        ),
        "reuse_category_adjustment_violation_count": reuse_adjustment_violations,
        "bundle_customer_consistency_passed": (
            None
            if not state.reservation_requests
            else (
                not shared_mismatches
                and storage_adjustment_violations == 0
                and reuse_adjustment_violations == 0
            )
        ),
        "snapshot_provenance_request_counts": dict(
            sorted(snapshot_provenance.items())
        ),
        "immutable_snapshot_requests_audited": int(
            snapshot_provenance.get("immutable_inquiry_v1", 0)
        ),
        "immutable_profile_snapshot_requests_audited": int(
            snapshot_provenance.get("immutable_profile_inquiry_v1", 0)
            + snapshot_provenance.get(
                "immutable_profile_inquiry_no_ceiling_v2", 0
            )
        ),
        "ceilingless_profile_snapshot_requests_audited": int(
            snapshot_provenance.get(
                "immutable_profile_inquiry_no_ceiling_v2", 0
            )
        ),
        "snapshot_invariant_violations": snapshot_violations,
        "stage_snapshot_invariants_passed": (
            None
            if not (
                snapshot_provenance.get("immutable_inquiry_v1", 0)
                or snapshot_provenance.get("immutable_profile_inquiry_v1", 0)
                or snapshot_provenance.get(
                    "immutable_profile_inquiry_no_ceiling_v2", 0
                )
            )
            else not snapshot_violations
        ),
    }


def _daily_rows(
    state: WorldState, models: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    models_by_day: dict[int, list[dict[str, Any]]] = defaultdict(list)
    tools_by_day: dict[int, list[dict[str, Any]]] = defaultdict(list)
    starts: Counter[int] = Counter(row.start_day for row in state.contracts)
    returns: Counter[int] = Counter(
        row.return_day for row in state.contracts if row.return_day <= state.max_days
    )
    for row in models:
        models_by_day[row["day"]].append(row)
    for row in tools:
        tools_by_day[row["day"]].append(row)
    result: list[dict[str, Any]] = []
    cumulative_revenue = 0.0
    cumulative_cost = 0.0
    cumulative_operating_pnl = 0.0
    cumulative_retail_liquidation_loss = 0.0
    legacy_rental_depreciation = _rental_depreciation_by_day(state)
    for report in state.daily_history:
        day = int(report["day"])
        finance = report["financial"]
        demand = report["demand"]
        end = report["end"]
        rental_revenue = _number(finance, "rental_revenue")
        retail_revenue = _number(finance, "retail_revenue")
        damage_deductions = _number(finance, "damage_deductions")
        revenue = rental_revenue + retail_revenue
        cost = sum(_number(finance, key) for key in COST_FIELDS)
        retail_cogs = _daily_retail_cogs(report)
        depreciation_field = next(
            (
                key
                for key in (
                    "rental_use_depreciation",
                    "rental_depreciation",
                    "rental_usage_depreciation",
                )
                if key in finance
            ),
            None,
        )
        rental_depreciation = (
            _number(finance, depreciation_field)
            if depreciation_field is not None
            else legacy_rental_depreciation.get(day, 0.0)
        )
        operating_revenue = revenue + damage_deductions
        operating_expense = (
            retail_cogs
            + _number(finance, "supplier_trial_handling_fee")
            + _number(finance, "supplier_trial_usage_fee")
            + _number(finance, "cleaning_cost")
            + _number(finance, "repair_cost")
            + _number(finance, "maintenance_cost")
            + _number(finance, "retail_holding_cost")
            + _number(finance, "rental_holding_cost")
            + _number(finance, "rental_idle_holding_cost")
            + _number(finance, "reservation_compensation")
            + _number(finance, "operating_fee")
            + rental_depreciation
        )
        operating_pnl = operating_revenue - operating_expense
        retail_liquidation_loss = _number(
            finance, "retail_liquidation_loss"
        )
        cumulative_revenue += revenue
        cumulative_cost += cost
        cumulative_operating_pnl += operating_pnl
        cumulative_retail_liquidation_loss += retail_liquidation_loss
        liquidation_value = end.get(
            "liquidation_business_value",
            end.get("business_value", end.get("net_worth")),
        )
        reconstructed_in_transit_book = sum(
            float(order.total_cost)
            for order in state.orders
            if order.placed_day <= day < order.delivery_day
        )
        book_net_assets = end.get(
            "book_net_assets",
            (
                float(end["cash"])
                - float(end["deposit_liability"])
                - float(end.get("unearned_revenue_liability", 0.0))
                - float(end.get("emergency_credit_liability", 0.0))
                + float(end["retail_inventory_book_value"])
                + float(end["rental_remaining_life_book_value"])
                + reconstructed_in_transit_book
            ),
        )
        going_concern_value = end.get("going_concern_business_value")
        operating_value = (
            going_concern_value
            if going_concern_value is not None
            else book_net_assets
        )
        calls = models_by_day[day]
        day_tools = tools_by_day[day]
        service_closed = int(
            demand.get("service_closed_rejections", demand.get("closed_rejections", 0))
        )
        capacity_policy = int(demand.get("capacity_policy_rejections", 0))
        horizon = int(demand.get("horizon_rejections", 0))
        row: dict[str, Any] = {
            "day": day,
            "weather": report["weather"],
            "calendar": report.get("calendar"),
            "customers": sum(int(value) for value in report["arrivals"].values()),
            **{f"arrivals_{key}": int(value) for key, value in report["arrivals"].items()},
            **{key: demand[key] for key in demand},
            **{key: finance[key] for key in finance},
            "retail_cogs": round(retail_cogs, 2),
            "rental_use_depreciation": round(rental_depreciation, 2),
            "rental_use_depreciation_basis": (
                "persisted_financial_ledger"
                if depreciation_field is not None
                else "contract_start_used_life_reconstruction"
            ),
            "operating_revenue": round(operating_revenue, 2),
            "operating_expense": round(operating_expense, 2),
            "daily_operating_pnl": round(operating_pnl, 2),
            "cumulative_operating_pnl": round(cumulative_operating_pnl, 2),
            "daily_business_pnl_after_liquidation": round(
                operating_pnl - retail_liquidation_loss, 2
            ),
            "cumulative_business_pnl_after_liquidation": round(
                cumulative_operating_pnl - cumulative_retail_liquidation_loss,
                2,
            ),
            "service_closed_rejections": service_closed,
            "capacity_policy_rejections": capacity_policy,
            "horizon_rejections": horizon,
            "eligible_requested_units": max(0, int(demand["requested_units"]) - horizon),
            "eligible_fulfillment_rate": (
                round(
                    int(demand["fulfilled_units"])
                    / max(1, int(demand["requested_units"]) - horizon),
                    6,
                )
                if int(demand["requested_units"]) - horizon > 0
                else None
            ),
            "operational_requested_units": (
                int(demand["fulfilled_units"])
                + int(demand.get("stockout_rejections", 0))
                + service_closed
                + capacity_policy
            ),
            "operational_fulfillment_rate": (
                round(
                    int(demand["fulfilled_units"])
                    / (
                        int(demand["fulfilled_units"])
                        + int(demand.get("stockout_rejections", 0))
                        + service_closed
                        + capacity_policy
                    ),
                    6,
                )
                if (
                    int(demand["fulfilled_units"])
                    + int(demand.get("stockout_rejections", 0))
                    + service_closed
                    + capacity_policy
                )
                > 0
                else None
            ),
            "asset_sale_proceeds": finance.get("asset_sale_proceeds", 0.0),
            "assets_sold": finance.get("assets_sold", 0),
            "opening_rental_holding_cost_basis": finance.get(
                "opening_rental_holding_cost_basis", 0.0
            ),
            "opening_rental_idle_holding_cost_basis": finance.get(
                "opening_rental_idle_holding_cost_basis", 0.0
            ),
            "revenue_total": round(revenue, 2),
            "cost_total": round(cost, 2),
            "cumulative_revenue": round(cumulative_revenue, 2),
            "cumulative_cost": round(cumulative_cost, 2),
            "cash_end": end["cash"],
            **(
                {
                    "initial_durable_acquisition_budget": end[
                        "durable_acquisition_budget"
                    ]["initial_durable_acquisition_budget"],
                    "used_durable_acquisition_budget_end": end[
                        "durable_acquisition_budget"
                    ]["used_durable_acquisition_budget"],
                    "committed_durable_acquisition_budget_end": end[
                        "durable_acquisition_budget"
                    ]["committed_durable_acquisition_budget"],
                    "remaining_durable_acquisition_budget_end": end[
                        "durable_acquisition_budget"
                    ]["remaining_durable_acquisition_budget"],
                }
                if "durable_acquisition_budget" in end
                else {}
            ),
            **(
                {
                    "initial_owned_durable_basis": end["durable_capital"][
                        "initial_owned_durable_basis"
                    ],
                    "additional_durable_capital_headroom": end[
                        "durable_capital"
                    ]["additional_durable_capital_headroom"],
                    "durable_capital_limit": end["durable_capital"][
                        "durable_capital_limit"
                    ],
                    "deployed_owned_durable_basis_end": end[
                        "durable_capital"
                    ]["deployed_owned_durable_basis"],
                    "committed_inbound_durable_basis_end": end[
                        "durable_capital"
                    ]["committed_inbound_durable_basis"],
                    "pending_release_next_day_end": end["durable_capital"][
                        "pending_release_next_day"
                    ],
                    "remaining_durable_capital_headroom_end": end[
                        "durable_capital"
                    ]["remaining_durable_capital_headroom"],
                }
                if "durable_capital" in end
                else {}
            ),
            "deposit_liability_end": end["deposit_liability"],
            "unearned_revenue_liability_end": end.get(
                "unearned_revenue_liability", 0.0
            ),
            "emergency_credit_liability_end": end.get(
                "emergency_credit_liability", 0.0
            ),
            "retail_units_end": end["retail_units"],
            "retail_book_value_end": end["retail_inventory_book_value"],
            "retail_salvage_value_end": end["retail_inventory_value"],
            "rental_book_value_end": end["rental_remaining_life_book_value"],
            "in_transit_book_value_end": round(
                reconstructed_in_transit_book, 2
            ),
            "rental_salvage_value_end": end.get(
                "rental_secondhand_value", end.get("rental_remaining_life_value", 0.0)
            ),
            "business_value_end": liquidation_value,
            "liquidation_business_value_end": liquidation_value,
            # The explicit names are the current reporting contract.  Keep the
            # historical business-value aliases above for downstream notebooks.
            "official_liquidation_value_end": liquidation_value,
            "going_concern_business_value_end": going_concern_value,
            "contract_adjusted_economic_value_end": (
                round(float(going_concern_value), 2)
                if going_concern_value is not None
                else None
            ),
            "book_net_assets_end": round(float(book_net_assets), 2),
            "operating_business_value_end": round(float(operating_value), 2),
            "operating_value_basis": (
                "going_concern_business_value"
                if going_concern_value is not None
                else "reconstructed_book_net_assets"
            ),
            "active_rentals_end": end["active_rentals"],
            "rental_utilization_end": end["rental_utilization"],
            "rental_starts": starts[day],
            "rental_returns": returns[day],
            "model_calls": len(calls),
            "input_tokens": sum(value["input_tokens"] for value in calls),
            "output_tokens": sum(value["output_tokens"] for value in calls),
            "tool_calls": len(day_tools),
            "tool_errors": sum(not value["ok"] for value in day_tools),
            "agent_tool_calls": sum(
                value.get("origin") == "agent" for value in day_tools
            ),
            "automatic_tool_calls": sum(
                value.get("origin") == "automatic" for value in day_tools
            ),
            "unknown_origin_tool_calls": sum(value.get('origin') == 'unknown' for value in day_tools),
            "identity_metrics_complete": all(value.get('identity_status') != 'unknown' for value in calls),
            "agent_tool_errors": sum(
                value.get("origin") == "agent" and not value["ok"]
                for value in day_tools
            ),
            "automatic_tool_errors": sum(
                value.get("origin") == "automatic" and not value["ok"]
                for value in day_tools
            ),
            "invalid_action_episodes": sum(
                value.get("invalid_action_episode", 0) for value in calls
            ),
            "generation_truncations": sum(
                value.get("generation_truncated", False) for value in calls
            ),
            "generation_truncated_tool_json": sum(
                bool(value.get("generation_truncated_tool_call_ids"))
                for value in calls
            ),
            "discarded_tool_calls": sum(value["discarded_tool_calls"] for value in calls),
            "read_only_tool_batches": sum(
                value["read_only_tool_batches"] for value in calls
            ),
            "batched_read_only_tool_calls": sum(
                value["batched_read_only_tool_calls"] for value in calls
            ),
            "rejected_tool_batches": sum(
                value["rejected_tool_batches"] for value in calls
            ),
            "rejected_batched_tool_calls": sum(
                value["rejected_batched_tool_calls"] for value in calls
            ),
        }
        result.append(row)
    return result


def _service_rows(state: WorldState, service_keys: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in state.daily_history:
        for service in service_keys:
            values = report["service_metrics"].get(service, {})
            rows.append(
                {
                    "day": report["day"],
                    "service_key": service,
                    "requested_units": values.get("requested_units", 0),
                    "fulfilled_units": values.get("fulfilled_units", 0),
                    "outside_rejections": values.get("outside_rejections", 0),
                    "stockout_rejections": values.get("stockout_rejections", 0),
                    "service_closed_rejections": values.get(
                        "service_closed_rejections", values.get("closed_rejections", 0)
                    ),
                    "capacity_policy_rejections": values.get(
                        "capacity_policy_rejections", 0
                    ),
                    "horizon_rejections": values.get("horizon_rejections", 0),
                    "revenue": values.get("revenue", 0.0),
                    "deposits_collected": values.get("deposits_collected", 0.0),
                }
            )
    return rows


def _service_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    money: dict[str, dict[str, float]] = defaultdict(lambda: {"revenue": 0.0, "deposits_collected": 0.0})
    for row in rows:
        service = row["service_key"]
        for key in (
            "requested_units",
            "fulfilled_units",
            "outside_rejections",
            "stockout_rejections",
            "service_closed_rejections",
            "capacity_policy_rejections",
            "horizon_rejections",
        ):
            totals[service][key] += int(row[key])
        money[service]["revenue"] += float(row["revenue"])
        money[service]["deposits_collected"] += float(row["deposits_collected"])
    return [
        {"service_key": service, **dict(totals[service]), **{key: round(value, 2) for key, value in money[service].items()}}
        for service in sorted(totals)
    ]


def _product_fulfillment_rows(
    environment: OutfitterEnvironment, state: WorldState
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in state.daily_history:
        day = int(report["day"])
        for service in environment.public_service_registry():
            service_key = service["service_key"]
            product_types = service["fulfillment_product_types"]
            metric = report["service_metrics"].get(service_key, {})
            breakdown = metric.get(
                "product_fulfillment_breakdown",
                metric.get("fulfillment_breakdown", {}),
            )
            for product_type in product_types:
                if product_type in breakdown:
                    fulfilled = int(breakdown[product_type].get("fulfilled_units", 0))
                    revenue = float(breakdown[product_type].get("revenue", 0.0))
                    rental_days = int(breakdown[product_type].get("rental_days", 0))
                    retail_cogs = float(breakdown[product_type].get("retail_cogs", 0.0))
                elif len(product_types) == 1:
                    fulfilled = int(metric.get("fulfilled_units", 0))
                    revenue = float(metric.get("revenue", 0.0))
                    rental_days = 0
                    retail_cogs = 0.0
                elif service["kind"] == "rental":
                    contracts = [
                        row
                        for row in state.contracts
                        if row.start_day == day and row.service_line == product_type
                    ]
                    fulfilled = len(contracts)
                    revenue = sum(float(row.rent_paid) for row in contracts)
                    rental_days = sum(int(row.duration_days) for row in contracts)
                    retail_cogs = 0.0
                else:
                    fulfilled = 0
                    revenue = 0.0
                    rental_days = 0
                    retail_cogs = 0.0
                rows.append(
                    {
                        "day": day,
                        "weather": report.get("weather"),
                        "service_key": service_key,
                        "product_type": product_type,
                        "kind": service["kind"],
                        "fulfilled_units": fulfilled,
                        "revenue": round(revenue, 2),
                        "rental_days": rental_days,
                        "retail_cogs": round(retail_cogs, 2),
                        "gross_contribution": round(revenue - retail_cogs, 2),
                    }
                )
    return rows


def _product_fulfillment_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {
            "fulfilled_units": 0,
            "revenue": 0.0,
            "rental_days": 0,
            "retail_cogs": 0.0,
            "gross_contribution": 0.0,
        }
    )
    for row in rows:
        key = (row["service_key"], row["product_type"])
        totals[key]["fulfilled_units"] += int(row["fulfilled_units"])
        totals[key]["revenue"] += float(row["revenue"])
        totals[key]["rental_days"] += int(row.get("rental_days", 0))
        totals[key]["retail_cogs"] += float(row.get("retail_cogs", 0.0))
        totals[key]["gross_contribution"] += float(
            row.get("gross_contribution", float(row["revenue"]) - float(row.get("retail_cogs", 0.0)))
        )
    return [
        {
            "service_key": service_key,
            "product_type": product_type,
            "fulfilled_units": int(values["fulfilled_units"]),
            "revenue": round(values["revenue"], 2),
            "rental_days": int(values["rental_days"]),
            "retail_cogs": round(values["retail_cogs"], 2),
            "gross_contribution": round(values["gross_contribution"], 2),
        }
        for (service_key, product_type), values in sorted(totals.items())
    ]


def _offer_fulfillment_rows(
    environment: OutfitterEnvironment, state: WorldState
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in state.daily_history:
        for service_key, metric in sorted(report["service_metrics"].items()):
            for offer_id, values in sorted(
                metric.get("offer_fulfillment_breakdown", {}).items()
            ):
                rows.append(
                    {
                        "day": report["day"],
                        "weather": report.get("weather"),
                        "service_key": service_key,
                        "product_type": values.get(
                            "product_type", environment.offers[offer_id].service_line
                        ),
                        "offer_id": offer_id,
                        "kind": environment.offers[offer_id].asset_kind,
                        "fulfilled_units": int(values.get("fulfilled_units", 0)),
                        "rental_days": int(values.get("rental_days", 0)),
                        "revenue": round(float(values.get("revenue", 0.0)), 2),
                        "deposits_collected": round(
                            float(values.get("deposits_collected", 0.0)), 2
                        ),
                        "retail_cogs": round(float(values.get("retail_cogs", 0.0)), 2),
                        "gross_contribution": round(
                            float(values.get("revenue", 0.0))
                            - float(values.get("retail_cogs", 0.0)),
                            2,
                        ),
                    }
                )
    return rows


def _offer_fulfillment_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
        lambda: {
            "fulfilled_units": 0,
            "rental_days": 0,
            "revenue": 0.0,
            "deposits_collected": 0.0,
            "retail_cogs": 0.0,
            "gross_contribution": 0.0,
        }
    )
    for row in rows:
        key = (row["service_key"], row["product_type"], row["offer_id"])
        for field in totals[key]:
            totals[key][field] += float(row[field])
    return [
        {
            "service_key": key[0],
            "product_type": key[1],
            "offer_id": key[2],
            "fulfilled_units": int(values["fulfilled_units"]),
            "rental_days": int(values["rental_days"]),
            "revenue": round(values["revenue"], 2),
            "deposits_collected": round(values["deposits_collected"], 2),
            "retail_cogs": round(values["retail_cogs"], 2),
            "gross_contribution": round(values["gross_contribution"], 2),
        }
        for key, values in sorted(totals.items())
    ]


def _product_weather_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate product contribution by realized weather, not forecast labels."""
    totals: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(
        lambda: {
            "fulfilled_units": 0.0,
            "rental_days": 0.0,
            "revenue": 0.0,
            "retail_cogs": 0.0,
            "gross_contribution": 0.0,
        }
    )
    for row in rows:
        key = (
            str(row.get("weather") or "unknown"),
            str(row["service_key"]),
            str(row["product_type"]),
            str(row.get("kind") or "unknown"),
        )
        for field in totals[key]:
            totals[key][field] += float(row.get(field, 0.0))
    return [
        {
            "weather": key[0],
            "service_key": key[1],
            "product_type": key[2],
            "kind": key[3],
            "fulfilled_units": int(values["fulfilled_units"]),
            "rental_days": int(values["rental_days"]),
            "revenue": round(values["revenue"], 2),
            "retail_cogs": round(values["retail_cogs"], 2),
            "gross_contribution": round(values["gross_contribution"], 2),
            "gross_margin_pct": (
                round(100 * values["gross_contribution"] / values["revenue"], 2)
                if values["revenue"]
                else None
            ),
        }
        for key, values in sorted(totals.items())
    ]


def _concentration_rows(
    rows: list[dict[str, Any]], *, level: str, operating_pnl: float
) -> list[dict[str, Any]]:
    """Return descriptive concentration and leave-one-out diagnostics.

    This is deliberately not a causal counterfactual: removing a SKU/service
    would change customer choices and costs.  It flags concentration using the
    observed gross contribution while keeping that limitation explicit in the
    column names and report prose.
    """
    if level not in {"offer", "service"}:
        raise ValueError(f"unsupported concentration level: {level}")
    key_fields = ("offer_id",) if level == "offer" else ("service_key",)
    totals: dict[tuple[str, ...], dict[str, Any]] = defaultdict(
        lambda: {
            "fulfilled_units": 0.0,
            "rental_days": 0.0,
            "revenue": 0.0,
            "retail_cogs": 0.0,
            "gross_contribution": 0.0,
        }
    )
    for row in rows:
        key = tuple(str(row[field]) for field in key_fields)
        values = totals[key]
        for field in (
            "fulfilled_units",
            "rental_days",
            "revenue",
            "retail_cogs",
            "gross_contribution",
        ):
            values[field] += float(row.get(field, 0.0))
        values.setdefault("service_key", row.get("service_key"))
        values.setdefault("product_type", row.get("product_type"))
        values.setdefault("kind", row.get("kind"))
    total_revenue = sum(values["revenue"] for values in totals.values())
    total_positive_contribution = sum(
        max(0.0, values["gross_contribution"]) for values in totals.values()
    )
    total_contribution = sum(values["gross_contribution"] for values in totals.values())
    result: list[dict[str, Any]] = []
    for key, values in totals.items():
        contribution = values["gross_contribution"]
        row: dict[str, Any] = {
            "level": level,
            key_fields[0]: key[0],
            "service_key": values["service_key"],
            "product_type": values["product_type"],
            "kind": values["kind"],
            "fulfilled_units": int(values["fulfilled_units"]),
            "rental_days": int(values["rental_days"]),
            "revenue": round(values["revenue"], 2),
            "retail_cogs": round(values["retail_cogs"], 2),
            "gross_contribution": round(contribution, 2),
            "revenue_share_pct": (
                round(100 * values["revenue"] / total_revenue, 2)
                if total_revenue
                else None
            ),
            "positive_gross_contribution_share_pct": (
                round(100 * max(0.0, contribution) / total_positive_contribution, 2)
                if total_positive_contribution
                else None
            ),
            "observed_leave_one_out_gross_contribution": round(
                total_contribution - contribution, 2
            ),
            "observed_leave_one_out_operating_pnl": round(
                operating_pnl - contribution, 2
            ),
        }
        result.append(row)
    return sorted(result, key=lambda row: (-row["gross_contribution"], str(row[key_fields[0]])))


def _standardized_model_cost(
    models: list[dict[str, Any]], config: OutfitterConfig, manifest: dict[str, Any]
) -> tuple[float, float, float]:
    """Use explicit standardized token prices when supplied, else run prices.

    The fallback makes historical reports exact replicas of their recorded
    token-cost convention.  Future configs can place the common rates in either
    ``config`` or ``environment_rules.economics`` without breaking old runs.
    """
    config_values = manifest.get("config", {})
    economics = manifest.get("environment_rules", {}).get("economics", {})
    def rate(name: str, fallback: float) -> float:
        for source in (economics, config_values):
            for key in (
                f"standardized_{name}_token_cost_per_million",
                f"standard_{name}_token_cost_per_million",
            ):
                if key in source:
                    return float(source[key])
        return fallback

    input_rate = rate("input", config.input_token_cost_per_million)
    output_rate = rate("output", config.output_token_cost_per_million)
    cost = (
        sum(row["input_tokens"] for row in models) * input_rate
        + sum(row["output_tokens"] for row in models) * output_rate
    ) / 1_000_000
    return round(cost, 6), input_rate, output_rate


def _model_cost_was_charged_to_cash(manifest: dict[str, Any]) -> bool:
    """Identify legacy executions where token cost reduced store cash."""
    values = manifest.get("config", {})
    economics = manifest.get("environment_rules", {}).get("economics", {})
    for source in (values, economics):
        if "charge_model_usage_to_cash" in source:
            return bool(source["charge_model_usage_to_cash"])
    # Before the switch was introduced, model calls always charged cash.
    return True


def _rental_activity_rows(
    environment: OutfitterEnvironment, state: WorldState
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "service_key": environment.service_key_for_product_type(
                str(row["product_type"])
            ),
        }
        for row in state.rental_activity
    ]


def _rental_activity_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter((row["product_type"], row["event_type"]) for row in rows)
    return [
        {"product_type": key[0], "event_type": key[1], "events": count}
        for key, count in sorted(counts.items())
    ]


def _purchased_rental_contribution(
    environment: OutfitterEnvironment,
    state: WorldState,
    *,
    completed_days: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Describe the realized contribution of assets acquired during the run.

    Revenue and asset-specific cash costs are exact. Holding cost is allocated
    from the public per-day unit-cost rule and is labelled approximate because
    legacy activity logs may not preserve every within-day status transition.
    This is a diagnostic decomposition, not a causal counterfactual.
    """

    purchased = [
        asset
        for asset in state.assets
        if int(asset.acquired_day) > 0 and asset.ownership_type == "owned"
    ]
    if not purchased:
        return [], [], []
    events_by_asset: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in state.rental_activity:
        events_by_asset[str(event["asset_id"])].append(event)
    contracts_by_asset: defaultdict[str, list[Any]] = defaultdict(list)
    for contract in state.contracts:
        contracts_by_asset[str(contract.asset_id)].append(contract)
    pending_cleaning = {
        str(job.asset_id): float(job.cleaning_cost) for job in state.cleaning_queue
    }
    asset_rows = []
    for asset in purchased:
        events = events_by_asset[asset.asset_id]
        contracts = contracts_by_asset[asset.asset_id]
        normal_returns = [
            row for row in events if row.get("event_type") == "returned_normal"
        ]
        damaged_returns = [
            row for row in events if row.get("event_type") == "returned_damaged"
        ]
        repair_starts = [
            row for row in events if row.get("event_type") == "repair_started"
        ]
        cleaning_cost = sum(
            float(row.get("scheduled_cleaning_cost", 0.0))
            for row in normal_returns
        )
        # A pending job is already represented by its returned_normal event in
        # current logs. This fallback covers older logs without that detail.
        if not normal_returns and asset.asset_id in pending_cleaning:
            cleaning_cost += pending_cleaning[asset.asset_id]
        repair_cost = sum(float(row.get("repair_cost", 0.0)) for row in repair_starts)
        maintenance_cost = sum(
            float(row.get("maintenance_cost", 0.0))
            for row in events
            if row.get("event_type") == "maintenance_started"
        )
        damage_deductions = sum(
            float(row.get("damage_deduction", 0.0)) for row in damaged_returns
        )
        billed_end_day = completed_days
        for event in events:
            event_type = str(event.get("event_type"))
            event_day = int(event.get("day", billed_end_day))
            if event_type in {"asset_sold", "retired_by_agent"}:
                billed_end_day = min(billed_end_day, event_day)
            elif event_type == "retired_service_life":
                billed_end_day = min(billed_end_day, event_day - 1)
        holding_days = max(0, billed_end_day - int(asset.acquired_day) + 1)
        if environment.config.lifecycle_economics_enabled:
            assert environment.lifecycle_scales is not None
            rental_days = sum(int(contract.duration_days) for contract in contracts)
            on_site_days = max(0, holding_days - rental_days)
            projection = environment._lifecycle_projection(asset.offer_id)
            capital_holding_cost = (
                holding_days
                * float(asset.unit_cost)
                * environment.lifecycle_scales.annual_capital_rate
                / 365.0
            )
            space_holding_cost = (
                on_site_days
                * float(projection["storage_burden_units"])
                * environment.lifecycle_scales.daily_space_rate_per_storage_unit
            )
            holding_cost = capital_holding_cost + space_holding_cost
        else:
            capital_holding_cost = (
                holding_days * RENTAL_HOLDING_RATE * float(asset.unit_cost)
            )
            space_holding_cost = 0.0
            holding_cost = capital_holding_cost
        if asset.status == "sold":
            disposition_value = float(asset.sale_proceeds or 0.0)
            disposition = "realized_sale_proceeds"
        else:
            quote = environment.secondhand_quote(asset, terminal=True)
            disposition_value = float(quote["offer_price"]) if quote is not None else 0.0
            disposition = "terminal_dealer_quote" if quote is not None else "zero"
        rental_revenue = sum(float(contract.rent_paid) for contract in contracts)
        contribution = (
            rental_revenue
            + damage_deductions
            + disposition_value
            - float(asset.unit_cost)
            - cleaning_cost
            - repair_cost
            - maintenance_cost
            - holding_cost
        )
        offer = environment.offers[asset.offer_id]
        asset_rows.append(
            {
                "offer_id": asset.offer_id,
                "service_key": environment.service_key_for_product_type(
                    asset.service_line
                ),
                "product_type": asset.service_line,
                "choice_group": offer.choice_group,
                "asset_id": asset.asset_id,
                "rental_contracts": len(contracts),
                "rental_days": sum(int(contract.duration_days) for contract in contracts),
                "rental_revenue": rental_revenue,
                "damage_deductions": damage_deductions,
                "disposition_value": disposition_value,
                "disposition_basis": disposition,
                "acquisition_cost": float(asset.unit_cost),
                "cleaning_cost": cleaning_cost,
                "repair_cost": repair_cost,
                "maintenance_cost": maintenance_cost,
                "holding_days_approx": holding_days,
                "holding_cost_approx": holding_cost,
                "capital_holding_cost_approx": capital_holding_cost,
                "space_holding_cost_approx": space_holding_cost,
                "realized_cash_recovery": round(
                    rental_revenue
                    + damage_deductions
                    + (float(asset.sale_proceeds or 0.0) if asset.status == "sold" else 0.0)
                    - cleaning_cost
                    - repair_cost
                    - maintenance_cost
                    - holding_cost,
                    2,
                ),
                "economic_recovery": round(
                    rental_revenue
                    + damage_deductions
                    + disposition_value
                    - cleaning_cost
                    - repair_cost
                    - maintenance_cost
                    - holding_cost,
                    2,
                ),
                "approx_contribution": contribution,
            }
        )

    def aggregate(keys: tuple[str, ...]) -> list[dict[str, Any]]:
        grouped: defaultdict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in asset_rows:
            grouped[tuple(row[key] for key in keys)].append(row)
        result = []
        numeric = (
            "rental_contracts",
            "rental_days",
            "rental_revenue",
            "damage_deductions",
            "disposition_value",
            "acquisition_cost",
            "cleaning_cost",
            "repair_cost",
            "maintenance_cost",
            "holding_days_approx",
            "holding_cost_approx",
            "capital_holding_cost_approx",
            "space_holding_cost_approx",
            "realized_cash_recovery",
            "economic_recovery",
            "approx_contribution",
        )
        for key, rows in grouped.items():
            value = {name: item for name, item in zip(keys, key)}
            value["purchased_assets"] = len(rows)
            for name in numeric:
                total = sum(float(row[name]) for row in rows)
                value[name] = int(total) if name in {
                    "rental_contracts",
                    "rental_days",
                    "holding_days_approx",
                } else round(total, 2)
            value["interpretation"] = (
                "descriptive purchased-asset cohort contribution; holding allocation "
                "is approximate and this is not a causal counterfactual"
            )
            acquisition = float(value["acquisition_cost"])
            value["realized_cash_recovery_ratio"] = (
                None
                if acquisition <= 0
                else round(float(value["realized_cash_recovery"]) / acquisition, 6)
            )
            value["economic_recovery_ratio"] = (
                None
                if acquisition <= 0
                else round(float(value["economic_recovery"]) / acquisition, 6)
            )
            value["unrecovered_economic_share"] = (
                None
                if acquisition <= 0
                else round(
                    max(0.0, 1.0 - float(value["economic_recovery"]) / acquisition),
                    6,
                )
            )
            result.append(value)
        return sorted(
            result,
            key=lambda row: (-float(row["approx_contribution"]),) + tuple(
                str(row[key]) for key in keys
            ),
        )

    by_offer = aggregate(("offer_id", "service_key", "product_type", "choice_group"))
    positive_total = sum(max(0.0, float(row["approx_contribution"])) for row in by_offer)
    flags = []
    for row in by_offer:
        share = (
            max(0.0, float(row["approx_contribution"])) / positive_total
            if positive_total > 0
            else 0.0
        )
        row["positive_contribution_share"] = round(share, 6)
        if share > 0.40:
            flags.append(str(row["offer_id"]))
    by_product = aggregate(("service_key", "product_type"))
    return by_offer, by_product, sorted(flags)


def _retail_rows(environment: OutfitterEnvironment, state: WorldState) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"units": 0, "book_value": 0.0, "terminal_value": 0.0}
    )
    for offer_id, inventory in state.retail_inventory.items():
        line = environment.offers[offer_id].service_line
        totals[line]["units"] += int(inventory["quantity"])
        totals[line]["book_value"] += float(inventory["quantity"]) * float(inventory["avg_unit_cost"])
        totals[line]["terminal_value"] += environment.retail_liquidation_proceeds(
            float(inventory["avg_unit_cost"]), int(inventory["quantity"])
        )
    return [
        {
            "service_key": environment.service_key_for_product_type(line),
            "product_type": line,
            "units": int(value["units"]),
            "book_value": round(value["book_value"], 2),
            "terminal_salvage_value": round(value["terminal_value"], 2),
        }
        for line, value in sorted(totals.items())
    ]


def _asset_rows(environment: OutfitterEnvironment, state: WorldState) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for asset in state.assets:
        grouped[(asset.service_line, asset.ownership_type, asset.status)].append(asset)
    return [
        {
            "service_key": environment.service_key_for_product_type(line),
            "product_type": line,
            "ownership_type": ownership_type,
            "status": status,
            "assets": len(values),
            "acquisition_cost": round(
                sum(
                    row.unit_cost
                    for row in values
                    if row.ownership_type == "owned"
                ),
                2,
            ),
            "catalog_unit_cost_reference": round(
                sum(row.unit_cost for row in values), 2
            ),
            "used_life_days": sum(row.used_life_days for row in values),
            "completed_rental_cycles": sum(
                row.used_rental_cycles for row in values
            ),
            "inspection_due_assets": sum(
                row.next_inspection_due_day is not None
                and state.day >= row.next_inspection_due_day
                for row in values
            ),
            "remaining_life_book_value": round(
                sum(
                    row.unit_cost * environment._remaining_life_ratio(row)
                    for row in values
                    if row.ownership_type == "owned"
                    and row.status not in {"retired", "sold"}
                ),
                2,
            ),
        }
        for (line, ownership_type, status), values in sorted(grouped.items())
    ]


def _supplier_trial_rows(
    environment: OutfitterEnvironment, state: WorldState
) -> list[dict[str, Any]]:
    assets = {asset.asset_id: asset for asset in state.assets}
    rows = []
    for trial in state.supplier_trials:
        contracts = [
            contract
            for contract in state.contracts
            if contract.supplier_trial_id == trial.trial_id
        ]
        asset = assets.get(str(trial.asset_id))
        rows.append(
            {
                **environment._public_supplier_trial(trial),
                "asset_status": None if asset is None else asset.status,
                "physically_held": (
                    False
                    if asset is None
                    else environment._warehouse_asset_held(asset)
                ),
                "contract_starts": len(contracts),
                "rental_days": sum(
                    int(contract.duration_days) for contract in contracts
                ),
                "rental_revenue": round(
                    sum(float(contract.rent_paid) for contract in contracts), 2
                ),
                "handling_fee": round(float(trial.handling_fee), 2),
                "usage_fees": round(
                    sum(
                        float(contract.supplier_trial_usage_fee)
                        for contract in contracts
                    ),
                    2,
                ),
                "terminal_value": 0.0,
            }
        )
    return rows


def _durable_capital_exit_rows(
    environment: OutfitterEnvironment, state: WorldState
) -> list[dict[str, Any]]:
    if not environment._v40_revolving_durable_capital():
        return []
    return [
        {
            "asset_id": asset.asset_id,
            "offer_id": asset.offer_id,
            "product_type": asset.service_line,
            "durable_capital_basis": environment._capital_money(
                asset.durable_capital_basis_cents
            ),
            "capital_exit_day": asset.capital_exit_day,
            "capital_exit_cause": asset.capital_exit_cause,
            "capital_release_effective_day": asset.capital_release_effective_day,
            "capital_status_at_analysis": environment._capital_asset_status(
                state, asset
            ),
        }
        for asset in sorted(state.assets, key=lambda row: row.asset_id)
        if asset.ownership_type == "owned" and asset.capital_exit_day is not None
    ]


def _order_rows(environment: OutfitterEnvironment, state: WorldState) -> list[dict[str, Any]]:
    return [
        {
            "order_id": row.order_id,
            "seller_id": row.seller_id,
            "placed_day": row.placed_day,
            "estimated_delivery_day": environment.estimated_order_delivery_day(row),
            "actual_delivery_day": row.delivery_day,
            "status": row.status,
            "total_cost": row.total_cost,
            "procurement_mode": row.procurement_mode,
            "supplier_trial_id": row.supplier_trial_id,
            "supplier_trial_handling_fee": row.supplier_trial_handling_fee,
            **(
                {
                    "order_durable_capital_basis": (
                        environment._durable_order_budget_amount(row)
                    )
                }
                if environment._v40_revolving_durable_capital()
                else {}
            ),
            "pack_count": sum(int(item["packs"]) for item in row.items),
            "unit_count": sum(int(item["packs"]) * int(item["package_units"]) for item in row.items),
            "service_keys": sorted(
                {
                    environment.service_key_for_product_type(
                        environment.offers[item["offer_id"]].service_line
                    )
                    for item in row.items
                }
            ),
            "product_types": sorted(
                {environment.offers[item["offer_id"]].service_line for item in row.items}
            ),
        }
        for row in state.orders
    ]


def _contract_rows(environment: OutfitterEnvironment, state: WorldState) -> list[dict[str, Any]]:
    return [
        {
            "contract_id": row.contract_id,
            "asset_id": row.asset_id,
            "offer_id": row.offer_id,
            "service_key": environment.service_key_for_product_type(row.service_line),
            "product_type": row.service_line,
            "start_day": row.start_day,
            "return_day": row.return_day,
            "reservation_id": row.reservation_id,
            "promised_return_day": row.promised_return_day,
            "approved_return_day": row.approved_return_day,
            "estimated_return_day": row.estimated_return_day,
            "actual_return_day": row.actual_return_day,
            "duration_days": row.duration_days,
            "rent_paid": row.rent_paid,
            "deposit": row.deposit,
            "damaged_on_return": row.damaged_on_return,
            "status": row.status,
            "supplier_trial_id": row.supplier_trial_id,
            "supplier_trial_usage_fee": row.supplier_trial_usage_fee,
        }
        for row in state.contracts
    ]


def _asset_sale_rows(environment: OutfitterEnvironment, state: WorldState) -> list[dict[str, Any]]:
    return [
        {
            "asset_id": row.asset_id,
            "offer_id": row.offer_id,
            "service_key": environment.service_key_for_product_type(row.service_line),
            "product_type": row.service_line,
            "sold_day": row.sold_day,
            "inspection_grade": row.sale_grade,
            "proceeds": row.sale_proceeds,
        }
        for row in state.assets
        if row.status == "sold"
    ]


def _retail_liquidation_rows(
    environment: OutfitterEnvironment, state: WorldState
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in state.daily_history:
        for item in report.get("agent_actions", {}).get(
            "retail_liquidations", []
        ):
            offer_id = str(item["offer_id"])
            offer = environment.offers[offer_id]
            rows.append(
                {
                    "day": int(report["day"]),
                    "offer_id": offer_id,
                    "service_key": environment.service_key_for_product_type(
                        offer.service_line
                    ),
                    "product_type": offer.service_line,
                    "name": offer.name,
                    "units": int(item.get("units", 0)),
                    "average_acquisition_unit_cost": float(
                        item.get("average_acquisition_unit_cost", 0.0)
                    ),
                    "dealer_unit_quote": float(
                        item.get("dealer_unit_quote", 0.0)
                    ),
                    "book_cost": float(item.get("book_cost", 0.0)),
                    "proceeds": float(item.get("proceeds", 0.0)),
                    "liquidation_loss": float(
                        item.get("liquidation_loss", 0.0)
                    ),
                }
            )
    return rows


def _reservation_analysis_rows(
    state: WorldState,
    *,
    extension_cash_audit: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    commitments = {row.request_id: row for row in state.reservation_commitments}
    events_by_commitment: defaultdict[str, list[Any]] = defaultdict(list)
    for event in state.reservation_events:
        if event.commitment_id:
            events_by_commitment[event.commitment_id].append(event)
    requests = []
    lines = []
    for request in state.reservation_requests:
        commitment = commitments.get(request.request_id)
        factorized_options_by_line = {
            str(row.get("line_id")): [
                dict(option)
                for option in row.get("options", [])
                if option.get("disposition") == "quote"
            ]
            for row in request.submitted_line_options
        }
        events = (
            []
            if commitment is None
            else events_by_commitment[commitment.commitment_id]
        )
        compensation = sum(
            float(event.details.get("compensation", 0.0)) for event in events
        )
        requested_service_unit_days = sum(
            int(line.service_units or line.customer_units)
            * line.duration_days
            for line in request.lines
        )
        accepted_service_unit_days = (
            requested_service_unit_days if commitment is not None else 0
        )
        committed_service_unit_days = (
            0
            if commitment is None
            else sum(
                int(line.service_units or line.customer_units)
                * line.duration_days
                for line in request.lines
                if line.status
                not in {
                    "optional_removed_at_quote",
                    "optional_removed_at_confirmation",
                }
            )
        )
        recovery_omitted_service_unit_days = 0
        if (
            commitment is not None
            and commitment.original_confirmed_plan_id is not None
            and commitment.selected_recovery_plan_id is not None
        ):
            current_line_ids = {line.line_id for line in commitment.lines}
            recovery_omitted_service_unit_days = sum(
                int(line.service_units or line.customer_units)
                * line.duration_days
                for line in request.lines
                if line.line_id not in current_line_ids
            )
            committed_service_unit_days = sum(
                int(line.service_units or line.customer_units)
                * line.duration_days
                for line in request.lines
                if line.line_id in current_line_ids
            )
        option_asset_days_by_line = [
            [
                int(option["asset_quantity"]) * line.duration_days
                for option in (
                    factorized_options_by_line.get(line.line_id)
                    or line.submitted_quote_options
                )
                if option.get("asset_quantity") is not None
            ]
            for line in request.lines
        ]
        option_asset_days_by_line = [
            values for values in option_asset_days_by_line if values
        ]
        duration_by_line = {
            line.line_id: int(line.duration_days) for line in request.lines
        }
        complete_contract_asset_days = [
            sum(
                int(line.get("asset_quantity") or 0)
                * duration_by_line.get(str(line.get("line_id")), 0)
                for line in contract.get("lines", [])
                if line.get("disposition") == "quote"
            )
            for contract in request.submitted_contract_options
        ]
        submitted_asset_days_min = (
            min(complete_contract_asset_days)
            if complete_contract_asset_days
            else sum(min(values) for values in option_asset_days_by_line)
            if option_asset_days_by_line
            else None
        )
        submitted_asset_days_max = (
            max(complete_contract_asset_days)
            if complete_contract_asset_days
            else sum(max(values) for values in option_asset_days_by_line)
            if option_asset_days_by_line
            else None
        )
        current_asset_lines = (
            commitment.lines
            if commitment is not None
            and commitment.original_confirmed_plan_id is not None
            else request.lines
        )
        cash_audit_fields: dict[str, Any] = {}
        if extension_cash_audit and commitment is not None:
            gross_rent_cash = round(
                float(commitment.prepayment_collected)
                + float(commitment.rent_collected_at_pickup)
                + float(commitment.post_pickup_rent_collected),
                2,
            )
            cash_audit_fields = {
                "prepayment_collected": float(
                    commitment.prepayment_collected
                ),
                "rent_due_at_pickup": float(
                    commitment.rent_due_at_pickup or 0.0
                ),
                "rent_collected_at_pickup": float(
                    commitment.rent_collected_at_pickup
                ),
                "post_pickup_rent_collected": float(
                    commitment.post_pickup_rent_collected
                ),
                "gross_rent_cash_collected": gross_rent_cash,
                "net_rent_cash_collected": round(
                    gross_rent_cash
                    - float(commitment.prepayment_refunded),
                    2,
                ),
            }
        requests.append(
            {
                "request_id": request.request_id,
                "commitment_id": request.commitment_id,
                "created_day": request.created_day,
                "decision_day": request.decision_day,
                "booking_lead_days": request.booking_lead_days,
                "original_pickup_day": request.pickup_day,
                "current_pickup_day": (
                    request.pickup_day
                    if commitment is None
                    else commitment.pickup_day
                ),
                "original_return_day": request.return_day,
                "promised_return_day": (
                    None if commitment is None else commitment.promised_return_day
                ),
                "approved_return_day": (
                    None if commitment is None else commitment.approved_return_day
                ),
                "estimated_return_day": (
                    None if commitment is None else commitment.estimated_return_day
                ),
                "actual_return_day": (
                    None if commitment is None else commitment.actual_return_day
                ),
                "request_status": request.status,
                "reservation_contract_mode": request.reservation_contract_mode,
                "promise_profile_schema_version": (
                    request.promise_profile_schema_version
                ),
                "profile_deposit_policy_id": request.profile_deposit_policy_id,
                "quote_action": request.quote_action,
                "quote_day": request.quote_day,
                "commitment_status": (
                    None if commitment is None else commitment.status
                ),
                "line_count": len(request.lines),
                "required_line_count": sum(line.required for line in request.lines),
                "optional_line_count": sum(not line.required for line in request.lines),
                "locked_rent": request.total_locked_rent,
                "quoted_line_count": sum(
                    line.quoted_daily_price is not None
                    or bool(line.submitted_quote_options)
                    or bool(factorized_options_by_line.get(line.line_id))
                    for line in request.lines
                )
                if not request.submitted_contract_options
                else len(
                    {
                        str(line.get("line_id"))
                        for contract in request.submitted_contract_options
                        for line in contract.get("lines", [])
                        if line.get("disposition") == "quote"
                    }
                ),
                "submitted_quote_option_count": sum(
                    len(
                        factorized_options_by_line.get(line.line_id)
                        or line.submitted_quote_options
                    )
                    for line in request.lines
                ),
                "submitted_contract_option_count": len(
                    request.submitted_contract_options
                ),
                "selected_contract_option_id": (
                    request.selected_contract_option_id
                ),
                "feedback_response_available": (
                    request.feedback_response_available
                ),
                "reported_contract_assessment_count": sum(
                    bool(row.get("reported_assessment", {}).get("available"))
                    for row in request.reported_contract_assessments
                ),
                "internal_plan_count": int(request.internal_plan_count),
                "acceptable_plan_count": int(request.acceptable_plan_count),
                "customer_authorized_plan_count": len(
                    request.customer_authorized_plans
                ),
                "customer_preferred_plan_id": request.customer_preferred_plan_id,
                "manager_confirmed_plan_id": request.manager_confirmed_plan_id,
                "plan_confirmation_origin": request.plan_confirmation_origin,
                "manager_confirmation_day": request.manager_confirmation_day,
                "decline_feedback_available": request.reported_decline_feedback.get(
                    "available"
                ),
                "decline_feedback_category": request.reported_decline_feedback.get(
                    "category"
                ),
                "decline_feedback_reported_concerns": "|".join(
                    str(row)
                    for row in request.reported_decline_feedback.get(
                        "reported_concerns", []
                    )
                ),
                "decline_feedback_customer_mentioned_line_id": (
                    request.reported_decline_feedback.get(
                        "customer_mentioned_line_id"
                    )
                ),
                "optional_lines_removed_at_quote": sum(
                    line.status == "optional_removed_at_quote"
                    for line in request.lines
                ),
                "prepayment_due": request.prepayment_due,
                "prepayment_refunded": round(
                    sum(
                        float(
                            event.details.get(
                                "prepayment_refund",
                                event.details.get("refund", 0.0),
                            )
                        )
                        for event in events
                    ),
                    2,
                ),
                "compensation": round(compensation, 2),
                **cash_audit_fields,
                "optional_line_credits": (
                    0.0
                    if commitment is None
                    else commitment.optional_line_credits
                ),
                "assignment_count": (
                    0 if commitment is None else len(commitment.assignments)
                ),
                "risk_exposed_day": (
                    None if commitment is None else commitment.risk_exposed_day
                ),
                "failure_cause": (
                    None if commitment is None else commitment.failure_cause
                ),
                "selected_or_committed_asset_days": sum(
                    line.quantity * line.duration_days
                    for line in current_asset_lines
                ),
                "submitted_option_asset_days_min": (
                    submitted_asset_days_min
                ),
                "submitted_option_asset_days_max": (
                    submitted_asset_days_max
                ),
                "requested_service_unit_days": requested_service_unit_days,
                "accepted_service_unit_days": accepted_service_unit_days,
                "committed_service_unit_days": committed_service_unit_days,
                "optional_removed_service_unit_days": (
                    accepted_service_unit_days - committed_service_unit_days
                ),
                "recovery_omitted_service_unit_days": (
                    recovery_omitted_service_unit_days
                ),
                "declined_or_expired_service_unit_days": (
                    requested_service_unit_days - accepted_service_unit_days
                ),
                "committed_asset_days": (
                    0
                    if commitment is None
                    else sum(
                        line.quantity * line.duration_days
                        for line in current_asset_lines
                        if line.status
                        not in {
                            "optional_removed_at_quote",
                            "optional_removed_at_confirmation",
                        }
                    )
                ),
            }
        )
        if request.pickup_date_options:
            offered_asset_days = [sum(int(l["asset_quantity"]) for l in p["lines"])
                                  * request.service_offer_terms["duration_days"]
                                  for p in request.customer_authorized_plans] if request.service_offer_terms else []
            requests[-1].update(request_kind="fixed_reward_offer" if request.service_offer_terms else "price_inquiry",
                pickup_date_options=copy.deepcopy(request.pickup_date_options),
                initial_agreed_pickup_day=commitment.initial_agreed_pickup_day if commitment else None,
                initial_agreed_return_day=commitment.initial_agreed_return_day if commitment else None,
                offered_asset_days_min=min(offered_asset_days) if offered_asset_days else None,
                offered_asset_days_max=max(offered_asset_days) if offered_asset_days else None)
        commitment_lines = (
            {line.line_id: line for line in commitment.lines}
            if commitment is not None
            else {}
        )
        for line in request.lines:
            current = commitment_lines.get(line.line_id, line)
            omitted_by_recovery = bool(
                commitment is not None
                and commitment.original_confirmed_plan_id is not None
                and line.line_id not in commitment_lines
                and commitment.selected_recovery_plan_id is not None
            )
            submitted_line_options = (
                factorized_options_by_line.get(line.line_id)
                or line.submitted_quote_options
            )
            option_asset_days = [
                int(option["asset_quantity"]) * line.duration_days
                for option in submitted_line_options
                if option.get("asset_quantity") is not None
            ]
            inquiry_snapshot_is_valid = request.latent_snapshot_provenance in {
                "immutable_inquiry_v1",
                "immutable_profile_inquiry_v1",
                "immutable_profile_inquiry_no_ceiling_v2",
                "legacy_inquiry_snapshot",
            }
            persisted_latent_ids = list(
                request.inquiry_latent_offer_ids_by_line.get(
                    line.line_id, [line.offer_id]
                )
            )
            persisted_latent_ceilings = dict(
                request.inquiry_latent_price_ceiling_by_line.get(
                    line.line_id, {}
                )
            )
            assignments = (
                []
                if commitment is None
                else [
                    assignment.asset_id
                    for assignment in commitment.assignments
                    if assignment.line_id == line.line_id
                ]
            )
            lines.append(
                {
                    "request_id": request.request_id,
                    "commitment_id": request.commitment_id,
                    "line_id": line.line_id,
                    "required": line.required,
                    "original_offer_id": (
                        line.candidate_offer_ids[0]
                        if line.candidate_offer_ids
                        else line.offer_id
                    ),
                    "current_offer_id": (
                        None if omitted_by_recovery else current.offer_id
                    ),
                    "legacy_exact_candidate_offer_ids": (
                        list(line.candidate_offer_ids)
                        if request.reservation_contract_mode
                        not in {
                            "promise_profiles_v1",
                            "adaptive_pairwise_exact_quotes_v1",
                            "complete_contract_options_v1",
                            "factorized_line_options_plan_set_v1",
                        }
                        else []
                    ),
                    "quoted_daily_price": line.quoted_daily_price,
                    "submitted_quote_options": [
                        dict(option) for option in submitted_line_options
                    ],
                    "quote_option_count": len(submitted_line_options),
                    "quote_option_offer_ids": [
                        str(option["offer_id"])
                        for option in submitted_line_options
                    ],
                    "quote_option_profile_ids": [
                        str(option["promise_profile_id"])
                        for option in submitted_line_options
                    ],
                    "quote_option_daily_prices": [
                        float(option["daily_unit_price"])
                        for option in submitted_line_options
                    ],
                    "quote_option_asset_days": option_asset_days,
                    "submitted_option_asset_days_min": (
                        min(option_asset_days) if option_asset_days else None
                    ),
                    "submitted_option_asset_days_max": (
                        max(option_asset_days) if option_asset_days else None
                    ),
                    "selected_quote_option_id": (
                        current.selected_quote_option_id
                    ),
                    "authorized_offer_ids": list(current.authorized_offer_ids),
                    "inquiry_latent_offer_ids": list(
                        persisted_latent_ids if inquiry_snapshot_is_valid else []
                    ),
                    "inquiry_latent_price_ceilings": dict(
                        persisted_latent_ceilings
                        if inquiry_snapshot_is_valid
                        else {}
                    ),
                    "legacy_mixed_offer_ids": list(
                        [] if inquiry_snapshot_is_valid else persisted_latent_ids
                    ),
                    "legacy_mixed_price_ceilings": dict(
                        {} if inquiry_snapshot_is_valid else persisted_latent_ceilings
                    ),
                    "authorized_price_ceilings": dict(
                        request.authorized_price_ceiling_by_line.get(
                            line.line_id, {}
                        )
                    ),
                    "recovery_offer_ids": list(
                        request.recovery_offer_ids_by_line.get(line.line_id, [])
                    ),
                    "recovery_price_ceilings": dict(
                        request.recovery_price_ceiling_by_line.get(
                            line.line_id, {}
                        )
                    ),
                    "latent_snapshot_provenance": (
                        request.latent_snapshot_provenance
                    ),
                    "selected_offer_id": (
                        None
                        if omitted_by_recovery
                        else current.selected_offer_id
                    ),
                    "staged_offer_id": current.staged_offer_id,
                    "fulfillment_mode": current.fulfillment_mode,
                    "reservation_contract_mode": (
                        request.reservation_contract_mode
                    ),
                    "promise_profile_id": current.promise_profile_id,
                    "profile_family_id": current.profile_family_id,
                    "profile_schema_version": current.profile_schema_version,
                    "profile_deposit_policy_id": (
                        request.profile_deposit_policy_id
                    ),
                    "service_line": current.service_line,
                    "service_key": current.service_key,
                    "quantity": 0 if omitted_by_recovery else current.quantity,
                    "asset_quantity": (
                        0 if omitted_by_recovery else current.asset_quantity
                    ),
                    "customer_units": current.customer_units,
                    "service_units": current.service_units,
                    "capacity_per_asset": current.capacity_per_asset,
                    "eligible_profile_ids": list(
                        line.eligible_profile_ids
                    ),
                    "eligible_profile_family_ids": list(
                        line.eligible_profile_family_ids
                    ),
                    "duration_days": current.duration_days,
                    "quote_duration_days": line.duration_days,
                    "selected_or_committed_asset_days": (
                        0
                        if omitted_by_recovery
                        else current.quantity * current.duration_days
                    ),
                    "requested_service_unit_days": (
                        int(line.service_units or line.customer_units)
                        * line.duration_days
                    ),
                    "accepted_service_unit_days": (
                        int(line.service_units or line.customer_units)
                        * line.duration_days
                        if commitment is not None
                        else 0
                    ),
                    "committed_service_unit_days": (
                        int(line.service_units or line.customer_units)
                        * line.duration_days
                        if commitment is not None
                        and line.status != "optional_removed_at_quote"
                        and not omitted_by_recovery
                        else 0
                    ),
                    "optional_removed_service_unit_days": (
                        int(line.service_units or line.customer_units)
                        * line.duration_days
                        if commitment is not None
                        and line.status == "optional_removed_at_quote"
                        else 0
                    ),
                    "recovery_omitted_service_unit_days": (
                        int(line.service_units or line.customer_units)
                        * line.duration_days
                        if omitted_by_recovery
                        else 0
                    ),
                    "declined_or_expired_service_unit_days": (
                        int(line.service_units or line.customer_units)
                        * line.duration_days
                        if commitment is None
                        else 0
                    ),
                    "locked_daily_price": (
                        0.0
                        if omitted_by_recovery
                        else current.locked_daily_price
                    ),
                    "locked_rent": (
                        0.0 if omitted_by_recovery else current.locked_rent
                    ),
                    "locked_security_deposit": (
                        0.0
                        if omitted_by_recovery
                        else current.locked_security_deposit
                    ),
                    "authorized_line_rent": (
                        None
                        if omitted_by_recovery
                        else current.authorized_line_rent
                    ),
                    "authorized_security_deposit": (
                        None
                        if omitted_by_recovery
                        else current.authorized_security_deposit
                    ),
                    "recovery_price_credit": (
                        0.0
                        if omitted_by_recovery
                        else current.recovery_price_credit
                    ),
                    "recovery_deposit_credit": (
                        0.0
                        if omitted_by_recovery
                        else current.recovery_deposit_credit
                    ),
                    "recovery_source_plan_id": (
                        commitment.selected_recovery_plan_id
                        if omitted_by_recovery
                        else current.recovery_source_plan_id
                    ),
                    "substitution_hint": current.substitution_hint,
                    # Line status records the quote/fulfillment stage.  Keep the
                    # parent commitment terminal state alongside it so a
                    # cancelled line is not mistaken for an active commitment.
                    "commitment_status": (
                        None if commitment is None else commitment.status
                    ),
                    "status": (
                        "omitted_by_recovery_plan"
                        if omitted_by_recovery
                        else current.status
                    ),
                    "credit_amount": current.credit_amount,
                    "assigned_asset_ids": assignments,
                }
            )
    events = [
        {
            "event_id": event.event_id,
            "day": event.day,
            "event_type": event.event_type,
            "request_id": event.request_id,
            "commitment_id": event.commitment_id,
            "details": event.details,
        }
        for event in state.reservation_events
    ]
    return requests, lines, events


_RISK_OPENING_ACTIVITY_TYPES = frozenset(
    {
        "asset_acquired",
        "cleaning_completed",
        "returned_normal",
        "returned_damaged",
        "retired_service_life",
        "repair_completed",
        "maintenance_completed",
        "maintenance_hold_started",
    }
)


def _classify_pickup_risk_blocker(
    required_by_offer: dict[str, int],
    asset_states: dict[str, dict[str, Any]],
    *,
    risk_day: int,
) -> tuple[str, str | None]:
    """Classify an exposure snapshot without asserting a causal counterfactual.

    Cleaning is proximate/sufficient only when its same-SKU units can cover the
    available-unit deficit.  If cleaning participates but cannot cover every
    deficit, the balance is `cleaning_and_other` (including structural absence).
    """

    by_offer: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in asset_states.values():
        if row.get("status") not in {"sold", "retired"}:
            by_offer[str(row["offer_id"])].append(row)
    shortage_rows: list[dict[str, Any]] = []
    for offer_id, required in sorted(required_by_offer.items()):
        assets = by_offer[offer_id]
        available = sum(row["status"] == "available" for row in assets)
        deficit = max(0, int(required) - available)
        if not deficit:
            continue
        cleaning_assets = [row for row in assets if row["status"] == "cleaning"]
        other_unavailable = sum(
            row["status"] not in {"available", "cleaning"} for row in assets
        )
        structural_absence = max(0, int(required) - len(assets))
        shortage_rows.append(
            {
                "deficit": deficit,
                "cleaning": cleaning_assets,
                "cleaning_contribution": min(deficit, len(cleaning_assets)),
                "other_involved": bool(other_unavailable or structural_absence),
            }
        )
    if not shortage_rows:
        # A runtime risk with no reproduced deficit should not be silently
        # labelled `other`; the caller records it as unclassified.
        return "unclassified", None
    cleaning_participates = any(row["cleaning_contribution"] for row in shortage_rows)
    cleaning_sufficient = all(
        len(row["cleaning"]) >= row["deficit"] for row in shortage_rows
    )
    other_involved = any(row["other_involved"] for row in shortage_rows)
    if cleaning_participates and cleaning_sufficient and not other_involved:
        blocker = "strict_cleaning_only"
    elif cleaning_participates:
        blocker = "cleaning_and_other"
    else:
        return "other", None

    # Exclusive timing label: same-day-return cleaning is used when those units
    # alone can account for each cleaning contribution; otherwise an earlier
    # backlog is necessary.  Unknown enqueue provenance remains explicit.
    timing_unknown = False
    needs_prior = False
    for row in shortage_rows:
        contribution = row["cleaning_contribution"]
        if not contribution:
            continue
        known = [asset.get("cleaning_since_day") for asset in row["cleaning"]]
        if any(value is None for value in known):
            timing_unknown = True
        same_day = sum(value == risk_day for value in known)
        if same_day < contribution:
            needs_prior = True
    timing = "unclassified" if timing_unknown else (
        "prior_day_backlog" if needs_prior else "same_day_return"
    )
    return blocker, timing


def _risk_asset_states_at_exposure(
    environment: OutfitterEnvironment,
    state: WorldState,
    *,
    risk_day: int,
    request_id: str,
    initial_asset_states: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Event-source the opening snapshot just before this sorted pickup attempt."""

    if initial_asset_states is None:
        opening = environment.reset()
        initial_asset_states = {
            asset.asset_id: {
                "asset_id": asset.asset_id,
                "offer_id": asset.offer_id,
                "status": asset.status,
                "cleaning_since_day": (
                    1 if asset.status == "cleaning" else None
                ),
            }
            for asset in opening.assets
        }
    statuses = copy.deepcopy(initial_asset_states)
    commitment_request = {
        row.commitment_id: row.request_id for row in state.reservation_commitments
    }
    risk_commitments_today = {
        str(event.commitment_id)
        for event in state.reservation_events
        if event.event_type == "pickup_risk_exposed"
        and int(event.day) == int(risk_day)
        and event.commitment_id is not None
    }

    def apply(event: dict[str, Any]) -> None:
        asset_id = str(event["asset_id"])
        row = statuses.setdefault(
            asset_id,
            {
                "asset_id": asset_id,
                "offer_id": str(event["offer_id"]),
                "status": "available",
                "cleaning_since_day": None,
            },
        )
        row["offer_id"] = str(event["offer_id"])
        row["status"] = str(event.get("status_after") or row["status"])
        if event["event_type"] == "returned_normal":
            row["cleaning_since_day"] = int(event["day"])
        elif row["status"] != "cleaning":
            row["cleaning_since_day"] = None

    for event in state.rental_activity:
        day = int(event["day"])
        event_type = str(event["event_type"])
        if day < risk_day:
            apply(event)
            continue
        if day > risk_day:
            continue
        opening_event = event_type in _RISK_OPENING_ACTIVITY_TYPES or (
            event_type == "asset_disposed"
            and event.get("disposal_reason") == "service_life_exhausted_on_return"
        )
        if opening_event:
            apply(event)
            continue
        if event_type != "rental_started" or not event.get("reservation_id"):
            continue
        commitment_id = str(event["reservation_id"])
        preceding_request = commitment_request.get(commitment_id)
        if (
            preceding_request is not None
            and preceding_request < request_id
            and commitment_id not in risk_commitments_today
        ):
            apply(event)
    return statuses


def _pickup_risk_attribution(
    environment: OutfitterEnvironment | None,
    state: WorldState,
) -> dict[str, Any]:
    risk_events = [
        event for event in state.reservation_events
        if event.event_type == "pickup_risk_exposed"
    ]
    proximate = Counter()
    timing = Counter()
    preceding = Counter()
    outcomes = Counter()
    events = list(state.reservation_events)
    event_index = {event.event_id: index for index, event in enumerate(events)}
    request_by_id = {request.request_id: request for request in state.reservation_requests}
    commitment_by_id = {
        commitment.commitment_id: commitment
        for commitment in state.reservation_commitments
    }
    confirmation_by_commitment: dict[str, Any] = {}
    for event in events:
        if (
            event.commitment_id
            and event.event_type == "reservation_confirmed_from_customer_plan_set"
        ):
            confirmation_by_commitment[str(event.commitment_id)] = event
    initial_asset_states = None
    if environment is not None and risk_events:
        opening = environment.reset()
        initial_asset_states = {
            asset.asset_id: {
                "asset_id": asset.asset_id,
                "offer_id": asset.offer_id,
                "status": asset.status,
                "cleaning_since_day": (
                    1 if asset.status == "cleaning" else None
                ),
            }
            for asset in opening.assets
        }

    for risk in risk_events:
        commitment_id = str(risk.commitment_id)
        risk_position = event_index[risk.event_id]
        before = events[:risk_position]
        after = events[risk_position + 1 :]
        preceding["accepted_reschedule"] += any(
            event.commitment_id == risk.commitment_id
            and event.event_type == "reschedule_accepted"
            for event in before
        )
        preceding["accepted_extension"] += any(
            event.commitment_id == risk.commitment_id
            and event.event_type == "extension_accepted"
            for event in before
        )
        following = [
            event for event in after
            if event.commitment_id == risk.commitment_id
        ]
        if any(event.event_type == "pickup_fulfilled" for event in following):
            outcomes["recovered_and_fulfilled"] += 1
        elif any(event.event_type == "cancelled_by_manager" for event in following):
            outcomes["manager_cancelled"] += 1
        elif any(event.event_type == "pickup_failed" for event in following):
            outcomes["pickup_failed"] += 1
        else:
            outcomes["unresolved_or_other"] += 1

        if environment is None:
            proximate["unclassified"] += 1
            timing["unclassified"] += 1
            continue
        commitment = commitment_by_id.get(commitment_id)
        request = request_by_id.get(str(risk.request_id))
        confirmation = confirmation_by_commitment.get(commitment_id)
        plan_id = (
            None
            if confirmation is None
            else confirmation.details.get("manager_confirmed_plan_id")
        )
        if plan_id is None and commitment is not None:
            plan_id = commitment.original_confirmed_plan_id
        plan = (
            None
            if request is None or plan_id is None
            else next(
                (
                    row for row in request.customer_authorized_plans
                    if str(row.get("plan_id")) == str(plan_id)
                ),
                None,
            )
        )
        if plan is None:
            proximate["unclassified"] += 1
            timing["unclassified"] += 1
            continue
        required = Counter()
        for line in plan.get("lines", []):
            if line.get("disposition") != "omit" and bool(line.get("required", True)):
                required[str(line["offer_id"])] += int(
                    line.get("asset_quantity") or line.get("quantity") or 0
                )
        if not required:
            proximate["unclassified"] += 1
            timing["unclassified"] += 1
            continue
        asset_states = _risk_asset_states_at_exposure(
            environment,
            state,
            risk_day=int(risk.day),
            request_id=str(risk.request_id),
            initial_asset_states=initial_asset_states,
        )
        blocker, timing_label = _classify_pickup_risk_blocker(
            dict(required), asset_states, risk_day=int(risk.day)
        )
        proximate[blocker] += 1
        if timing_label is not None:
            timing[timing_label] += 1

    return {
        "exposures": len(risk_events),
        "proximate_blockers": {
            key: int(proximate[key])
            for key in (
                "strict_cleaning_only",
                "cleaning_and_other",
                "other",
                "unclassified",
            )
        },
        "cleaning_timing": {
            key: int(timing[key])
            for key in (
                "same_day_return",
                "prior_day_backlog",
                "unclassified",
            )
        },
        "preceding_changes": {
            key: int(preceding[key])
            for key in ("accepted_reschedule", "accepted_extension")
        },
        "outcomes": {
            key: int(outcomes[key])
            for key in (
                "recovered_and_fulfilled",
                "manager_cancelled",
                "pickup_failed",
                "unresolved_or_other",
            )
        },
        "interpretation": (
            "Proximate physical gap at risk exposure only. Preceding accepted "
            "changes are reported separately and are not claimed to have caused the risk."
        ),
        "selected_plan_basis": (
            "original manager-confirmed customer-authorized plan at exposure; "
            "never final post-recovery commitment lines"
        ),
    }


def _reservation_summary(
    state: WorldState,
    rows: list[dict[str, Any]],
    line_rows: list[dict[str, Any]] | None = None,
    *,
    environment: OutfitterEnvironment | None = None,
) -> dict[str, Any]:
    from .service_commitments import service_commitment_diagnostics
    from .reservation_value import _contingency_diagnostics
    v32 = any(r.pickup_date_options for r in state.reservation_requests)
    price_inquiry_count = sum(not r.service_offer_terms for r in state.reservation_requests)
    line_rows = line_rows or []
    statuses = Counter(str(row["request_status"]) for row in rows)
    confirmed = int(state.cumulative_reservations_confirmed)
    pickups = int(state.cumulative_reservation_pickups)
    selected_or_committed_asset_days = sum(
        int(row["selected_or_committed_asset_days"]) for row in rows
    )
    accepted_asset_days = sum(
        int(row["committed_asset_days"])
        for row in rows
        if row["commitment_id"] is not None
    )
    accepted_lines = [
        row
        for row in line_rows
        if row.get("commitment_id") is not None
        and row.get("status")
        not in {
            "optional_removed_at_quote",
            "optional_removed_at_confirmation",
        }
        and row.get("authorized_offer_ids")
    ]
    authorization_sizes = Counter(
        len(row["authorized_offer_ids"]) for row in accepted_lines
    )
    legacy_candidate_sizes = Counter(
        len(row.get("legacy_exact_candidate_offer_ids") or [])
        for row in line_rows
        if row.get("reservation_contract_mode")
        not in {
            "promise_profiles_v1",
            "adaptive_pairwise_exact_quotes_v1",
            "complete_contract_options_v1",
            "factorized_line_options_plan_set_v1",
        }
    )
    eligible_profile_sizes = Counter(
        len(row.get("eligible_profile_ids") or [])
        for row in line_rows
        if row.get("reservation_contract_mode")
        in {
            "promise_profiles_v1",
            "adaptive_pairwise_exact_quotes_v1",
            "complete_contract_options_v1",
            "factorized_line_options_plan_set_v1",
        }
        and row.get("eligible_profile_ids")
    )
    inquiry_latent_sizes = Counter(
        len(row["inquiry_latent_offer_ids"])
        for row in line_rows
        if row.get("inquiry_latent_offer_ids")
    )
    legacy_mixed_sizes = Counter(
        len(row["legacy_mixed_offer_ids"])
        for row in line_rows
        if row.get("legacy_mixed_offer_ids")
    )
    recovery_sizes = Counter(
        len(row.get("recovery_offer_ids") or [])
        for row in accepted_lines
    )
    snapshot_provenance = Counter(
        str(row.get("latent_snapshot_provenance") or "unknown")
        for row in line_rows
    )
    fulfillment_modes = Counter(
        str(row["fulfillment_mode"])
        for row in line_rows
        if row.get("fulfillment_mode")
    )
    contract_modes = Counter(
        str(row.get("reservation_contract_mode") or "exact_candidates_v15")
        for row in rows
    )
    profile_selections = Counter(
        str(row["promise_profile_id"])
        for row in line_rows
        if row.get("promise_profile_id")
    )
    profile_families = Counter(
        str(row["profile_family_id"])
        for row in line_rows
        if row.get("profile_family_id")
    )
    deposit_policies = Counter(
        str(row["profile_deposit_policy_id"])
        for row in rows
        if row.get("profile_deposit_policy_id")
    )
    pairwise_lines = [
        row
        for row in line_rows
        if row.get("reservation_contract_mode")
        == "adaptive_pairwise_exact_quotes_v1"
        and row.get("quote_option_count")
    ]
    pairwise_menu_sizes = Counter(
        int(row["quote_option_count"]) for row in pairwise_lines
    )
    pairwise_exposures = Counter(
        offer_id
        for row in pairwise_lines
        for offer_id in row.get("quote_option_offer_ids", [])
    )
    pairwise_selections = Counter(
        str(row["current_offer_id"])
        for row in pairwise_lines
        if row.get("commitment_id") is not None
        and row.get("status") != "optional_removed_at_quote"
        and row.get("selected_quote_option_id")
    )
    pairwise_selection_positions = Counter(
        int(str(row["selected_quote_option_id"]).removeprefix("option-"))
        for row in pairwise_lines
        if row.get("selected_quote_option_id")
    )
    pairwise_price_spreads = [
        round(max(prices) - min(prices), 2)
        for row in pairwise_lines
        if len(prices := list(row.get("quote_option_daily_prices") or [])) == 2
    ]
    pairwise_requests = [
        row
        for row in rows
        if row.get("reservation_contract_mode")
        == "adaptive_pairwise_exact_quotes_v1"
    ]
    complete_contract_requests = [
        request
        for request in state.reservation_requests
        if request.reservation_contract_mode == "complete_contract_options_v1"
    ]
    complete_contract_option_counts = Counter(
        len(request.submitted_contract_options)
        for request in complete_contract_requests
        if request.quote_action == "quote"
    )
    complete_contract_selected_positions = Counter(
        int(str(request.selected_contract_option_id).removeprefix("contract-option-"))
        for request in complete_contract_requests
        if request.selected_contract_option_id
    )
    complete_contract_offer_exposures = Counter(
        str(line["offer_id"])
        for request in complete_contract_requests
        for contract in request.submitted_contract_options
        for line in contract.get("lines", [])
        if line.get("disposition") == "quote" and line.get("offer_id")
    )
    complete_contract_optional_dispositions = Counter(
        str(line.get("disposition"))
        for request in complete_contract_requests
        for contract in request.submitted_contract_options
        for line in contract.get("lines", [])
        if not next(
            row.required
            for row in request.lines
            if row.line_id == str(line.get("line_id"))
        )
    )
    assessment_categories = Counter(
        str(assessment["reported_assessment"]["category"])
        for request in complete_contract_requests
        for assessment in request.reported_contract_assessments
        if assessment.get("reported_assessment", {}).get("available")
    )
    assessment_outcomes = Counter(
        str(assessment.get("outcome") or "unknown")
        for request in complete_contract_requests
        for assessment in request.reported_contract_assessments
    )
    feedback_audit_rows = [
        dict(row)
        for request in complete_contract_requests
        for row in request.contract_feedback_audit.get("rows", [])
    ]
    feedback_latent_topics = Counter(
        str(row.get("latent_topic") or "unknown") for row in feedback_audit_rows
    )
    feedback_alignment_by_category: dict[str, dict[str, Any]] = {}
    for category in sorted(
        {str(row.get("reported_category")) for row in feedback_audit_rows}
    ):
        category_rows = [
            row
            for row in feedback_audit_rows
            if str(row.get("reported_category")) == category
        ]
        feedback_alignment_by_category[category] = {
            "count": len(category_rows),
            "mean_price_gain": round(
                sum(float(row.get("price_gain", 0.0)) for row in category_rows)
                / len(category_rows),
                6,
            ),
            "mean_configuration_gain": round(
                sum(
                    float(row.get("configuration_gain", 0.0))
                    for row in category_rows
                )
                / len(category_rows),
                6,
            ),
            "price_action_reversal_share": round(
                sum(
                    float(row.get("contract_score", 0.0))
                    + float(row.get("price_gain", 0.0))
                    > float(row.get("reference_score", 0.0))
                    for row in category_rows
                )
                / len(category_rows),
                6,
            ),
            "configuration_action_reversal_share": round(
                sum(
                    float(row.get("contract_score", 0.0))
                    + float(row.get("configuration_gain", 0.0))
                    > float(row.get("reference_score", 0.0))
                    for row in category_rows
                )
                / len(category_rows),
                6,
            ),
        }
    factorized_requests = [
        request
        for request in state.reservation_requests
        if request.reservation_contract_mode
        == "factorized_line_options_plan_set_v1" and not request.service_offer_terms
    ]
    factorized_internal_plan_counts = Counter(
        int(request.internal_plan_count)
        for request in factorized_requests
        if request.quote_action == "quote"
    )
    factorized_authorized_plan_counts = Counter(
        len(request.customer_authorized_plans)
        for request in factorized_requests
        if request.quote_action == "quote"
    )
    factorized_confirmation_origins = Counter(
        str(request.plan_confirmation_origin)
        for request in factorized_requests
        if request.plan_confirmation_origin
    )
    factorized_confirmation_preferences = Counter(
        "customer_preferred"
        if request.manager_confirmed_plan_id == request.customer_preferred_plan_id
        else "authorized_alternative"
        for request in factorized_requests
        if request.manager_confirmed_plan_id is not None
    )
    factorized_decline_feedback_categories = Counter(
        str(request.reported_decline_feedback["category"])
        for request in factorized_requests
        if request.reported_decline_feedback.get("available")
        and "category" in request.reported_decline_feedback
    )
    factorized_decline_feedback_tags = Counter(
        str(tag)
        for request in factorized_requests
        if request.reported_decline_feedback.get("available")
        for tag in request.reported_decline_feedback.get(
            "reported_concerns", []
        )
    )
    factorized_decline_feedback_concern_sets = Counter(
        "+".join(
            str(tag)
            for tag in request.reported_decline_feedback.get(
                "reported_concerns", []
            )
        )
        for request in factorized_requests
        if request.reported_decline_feedback.get("available")
        and request.reported_decline_feedback.get("reported_concerns")
    )
    factorized_line_mentions = [
        request
        for request in factorized_requests
        if request.reported_decline_feedback.get(
            "customer_mentioned_line_id"
        )
        is not None
    ]
    factorized_line_mention_service_counts: Counter[str] = Counter()
    factorized_line_mention_scope_violations = 0
    for request in factorized_requests:
        feedback = request.reported_decline_feedback
        mentioned_line_id = feedback.get("customer_mentioned_line_id")
        if mentioned_line_id is None:
            continue
        line_by_id = {line.line_id: line for line in request.lines}
        mentioned_line = line_by_id.get(str(mentioned_line_id))
        if mentioned_line is not None:
            factorized_line_mention_service_counts[
                str(mentioned_line.service_key)
            ] += 1
        if (
            request.customer_authorized_plans
            or feedback.get("available") is not True
            or (
                feedback.get("category") != EQUIPMENT_FIT_CONCERN
                and EQUIPMENT_CONFIGURATION_TAG
                not in feedback.get("reported_concerns", [])
            )
            or mentioned_line is None
        ):
            factorized_line_mention_scope_violations += 1
    factorized_feedback_audit_rows = [
        request.plan_set_feedback_audit
        for request in factorized_requests
        if "price_burden" in request.plan_set_feedback_audit
    ]
    factorized_feedback_alignment: dict[str, dict[str, Any]] = {}
    for category in sorted(
        {
            str(row.get("reported_category"))
            for row in factorized_feedback_audit_rows
        }
    ):
        category_rows = [
            row
            for row in factorized_feedback_audit_rows
            if str(row.get("reported_category")) == category
        ]
        factorized_feedback_alignment[category] = {
            "count": len(category_rows),
            "mean_price_burden": round(
                sum(float(row.get("price_burden", 0.0)) for row in category_rows)
                / len(category_rows),
                6,
            ),
            "mean_equipment_fit_regret": round(
                sum(
                    float(row.get("equipment_fit_regret", 0.0))
                    for row in category_rows
                )
                / len(category_rows),
                6,
            ),
        }
    factorized_tag_audit_rows = [
        request.plan_set_feedback_audit
        for request in factorized_requests
        if "price_action_gain" in request.plan_set_feedback_audit
        and request.plan_set_feedback_audit.get("response_available") is True
    ]
    factorized_tag_action_alignment: dict[str, dict[str, Any]] = {}
    for tag, gain_key in (
        ("price_value", "price_action_gain"),
        ("equipment_configuration", "equipment_action_gain"),
    ):
        tagged = [
            row
            for row in factorized_tag_audit_rows
            if tag in row.get("reported_concerns", [])
        ]
        untagged = [
            row
            for row in factorized_tag_audit_rows
            if tag not in row.get("reported_concerns", [])
        ]
        tagged_mean = (
            sum(float(row[gain_key]) for row in tagged) / len(tagged)
            if tagged
            else None
        )
        untagged_mean = (
            sum(float(row[gain_key]) for row in untagged) / len(untagged)
            if untagged
            else None
        )
        factorized_tag_action_alignment[tag] = {
            "tagged_count": len(tagged),
            "untagged_count": len(untagged),
            "tagged_mean_action_gain": (
                None if tagged_mean is None else round(tagged_mean, 6)
            ),
            "untagged_mean_action_gain": (
                None if untagged_mean is None else round(untagged_mean, 6)
            ),
            "mean_action_gain_lift": (
                None
                if tagged_mean is None or untagged_mean is None
                else round(tagged_mean - untagged_mean, 6)
            ),
        }
    factorized_multi_plan_requests = [
        request
        for request in factorized_requests
        if len(request.customer_authorized_plans) >= 2
    ]
    factorized_max_pairwise_line_differences: list[int] = []
    for request in factorized_multi_plan_requests:
        signatures = []
        for plan in request.customer_authorized_plans:
            signatures.append(
                {
                    str(line.get("line_id")): (
                        str(line.get("disposition")),
                        str(line.get("offer_id")),
                        str(line.get("promise_profile_id")),
                        int(line.get("asset_quantity") or 0),
                    )
                    for line in plan.get("lines", [])
                }
            )
        factorized_max_pairwise_line_differences.append(
            max(
                sum(
                    left.get(line_id) != right.get(line_id)
                    for line_id in set(left) | set(right)
                )
                for index, left in enumerate(signatures)
                for right in signatures[index + 1 :]
            )
        )
    unaccepted_pairwise_quoted_lines = [
        row
        for row in pairwise_lines
        if row.get("commitment_id") is None
        and row.get("submitted_option_asset_days_min") is not None
        and row.get("submitted_option_asset_days_max") is not None
    ]
    manager_cancellations = int(statuses.get("cancelled_by_manager", 0))
    commitments_by_id = {
        commitment.commitment_id: commitment
        for commitment in state.reservation_commitments
    }
    manager_cancellation_events = [
        event
        for event in state.reservation_events
        if event.event_type == "cancelled_by_manager"
    ]
    manager_cancellation_origin_counts = Counter(
        str(
            event.details.get("cancellation_origin")
            or (
                "risk_alert"
                if event.details.get("cause") == "early_risk_cancellation"
                else "proactive"
            )
        )
        for event in manager_cancellation_events
    )
    manager_cancellation_notice_distribution = Counter(
        int(event.details.get("notice_days", 0))
        for event in manager_cancellation_events
    )

    def affected_service_unit_days(event: Any) -> int:
        if "required_service_unit_days" in event.details:
            return int(event.details["required_service_unit_days"])
        commitment = commitments_by_id.get(str(event.commitment_id))
        if commitment is None:
            return 0
        return sum(
            int(line.service_units or line.customer_units)
            * int(line.duration_days)
            for line in commitment.lines
            if line.required
        )

    pickup_resolutions = pickups + int(state.cumulative_reservation_pickup_failures)
    event_types = Counter(event.event_type for event in state.reservation_events)
    reschedule_counts = Counter(
        change.commitment_id
        for change in state.reservation_change_requests
        if change.change_type == "reschedule"
    )
    request_by_id = {
        request.request_id: request for request in state.reservation_requests
    }

    def reschedule_tail(commitments: list[Any]) -> dict[str, Any]:
        counts = [
            int(reschedule_counts.get(commitment.commitment_id, 0))
            for commitment in commitments
        ]
        denominator = len(counts)
        return {
            "commitments": denominator,
            "request_count_distribution": dict(
                sorted(Counter(counts).items())
            ),
            "at_least_one": sum(value >= 1 for value in counts),
            "at_least_two": sum(value >= 2 for value in counts),
            "at_least_three": sum(value >= 3 for value in counts),
            "p_at_least_one": (
                round(sum(value >= 1 for value in counts) / denominator, 6)
                if denominator
                else None
            ),
            "p_at_least_two": (
                round(sum(value >= 2 for value in counts) / denominator, 6)
                if denominator
                else None
            ),
            "p_at_least_three": (
                round(sum(value >= 3 for value in counts) / denominator, 6)
                if denominator
                else None
            ),
            "maximum": max(counts, default=0),
        }

    lead_bands = {
        "1-3": (1, 3),
        "4-7": (4, 7),
        "8-10": (8, 10),
        "11-14": (11, 14),
    }
    reschedule_tail_summary = reschedule_tail(
        list(state.reservation_commitments)
    )
    reschedule_tail_summary["by_booking_lead"] = {
        label: reschedule_tail(
            [
                commitment
                for commitment in state.reservation_commitments
                if low
                <= int(request_by_id[commitment.request_id].booking_lead_days)
                <= high
            ]
        )
        for label, (low, high) in lead_bands.items()
    }
    pickup_failure_events = [
        event
        for event in state.reservation_events
        if event.event_type == "pickup_failed"
    ]
    reputation_snapshots = [
        dict(request.service_reputation_snapshot)
        for request in state.reservation_requests
        if request.service_reputation_snapshot.get("enabled")
    ]
    reputation_shadow_flips = 0
    reputation_evaluable_requests = 0
    for request in state.reservation_requests:
        snapshot = request.service_reputation_snapshot
        audit = request.plan_set_feedback_audit
        if (
            not snapshot.get("enabled")
            or not audit.get("plan_scores")
        ):
            continue
        penalty = float(snapshot.get("customer_choice_penalty", 0.0))
        best_with = max(float(value) for value in audit["plan_scores"].values())
        outside = max(
            float(audit.get("outside_score", float("-inf"))),
            float(audit.get("abandon_score", float("-inf"))),
        )
        reputation_evaluable_requests += 1
        reputation_shadow_flips += (
            best_with < outside <= best_with + penalty
        )
    pickup_fulfilled_events = [
        event
        for event in state.reservation_events
        if event.event_type == "pickup_fulfilled"
    ]
    contingency_diagnostics = _contingency_diagnostics(
        state, completed_day=len(state.daily_history)
    )
    decision_counts = Counter()
    for row in rows:
        action = row.get("quote_action")
        status = str(row.get("request_status"))
        if action == "quote":
            decision_counts["explicit_quote"] += 1
        elif status == "declined_by_manager":
            decision_counts["decline"] += 1
        elif status == "expired_unquoted":
            decision_counts["expired_unquoted"] += 1
        elif action:
            # Old V9 shortcut actions remain identifiable without reviving their
            # reference-price fields in the current state or report schema.
            decision_counts["legacy_non_explicit_quote"] += 1
    cash_audit_rows = [
        row for row in rows if "gross_rent_cash_collected" in row
    ]
    pickup_risk_attribution = _pickup_risk_attribution(environment, state)
    return {
        "requests": len(rows),
        **({"service_commitments": service_commitment_diagnostics(state),
            "quote_rate_denominator": "price_inquiries_only"} if v32 else {}),
        "quotes": state.cumulative_reservation_quotes,
        "decision_counts": dict(sorted(decision_counts.items())),
        "quote_rate": round(
            state.cumulative_reservation_quotes / (price_inquiry_count if v32 else len(rows)), 6
        )
        if (price_inquiry_count if v32 else len(rows))
        else None,
        "customer_quote_declines": (
            state.cumulative_reservation_customer_quote_declines
        ),
        "manager_quote_declines": (
            state.cumulative_reservation_manager_quote_declines
        ),
        "optional_lines_removed_at_quote": (
            state.cumulative_reservation_optional_lines_removed_at_quote
        ),
        "status_counts": dict(sorted(statuses.items())),
        "confirmed": confirmed,
        "confirmation_rate": round(confirmed / len(rows), 6) if rows else None,
        "pickups": pickups,
        "pickup_rate_given_confirmation": (
            round(pickups / confirmed, 6) if confirmed else None
        ),
        "pickup_failures": state.cumulative_reservation_pickup_failures,
        "pickup_failure_cause_counts": dict(
            sorted(
                Counter(
                    str(event.details.get("cause") or "unknown")
                    for event in pickup_failure_events
                ).items()
            )
        ),
        "pickup_failure_days": [
            int(event.day) for event in pickup_failure_events
        ],
        "pickup_failures_after_decision_horizon": sum(
            int(event.day) > int(state.decision_horizon_day)
            for event in pickup_failure_events
        ),
        "pickups_after_decision_horizon": sum(
            int(event.day) > int(state.decision_horizon_day)
            for event in pickup_fulfilled_events
        ),
        "pickup_resolution_count": pickup_resolutions,
        "pickup_success_rate_given_pickup_resolution": (
            round(pickups / pickup_resolutions, 6)
            if pickup_resolutions
            else None
        ),
        "manager_cancellations": manager_cancellations,
        "proactive_manager_cancellations": int(
            manager_cancellation_origin_counts.get("proactive", 0)
        ),
        "risk_alert_manager_cancellations": int(
            manager_cancellation_origin_counts.get("risk_alert", 0)
        ),
        "same_day_manager_cancellations": sum(
            int(event.details.get("notice_days", 0)) == 0
            for event in manager_cancellation_events
        ),
        "manager_cancellation_origin_counts": dict(
            sorted(manager_cancellation_origin_counts.items())
        ),
        "manager_cancellation_notice_distribution": dict(
            sorted(manager_cancellation_notice_distribution.items())
        ),
        "manager_caused_required_service_unit_days": sum(
            affected_service_unit_days(event)
            for event in (
                manager_cancellation_events + pickup_failure_events
            )
        ),
        "manager_cancellation_required_service_unit_days": sum(
            affected_service_unit_days(event)
            for event in manager_cancellation_events
        ),
        "pickup_failure_required_service_unit_days": sum(
            affected_service_unit_days(event)
            for event in pickup_failure_events
        ),
        "direct_manager_failure_compensation": round(
            sum(
                float(event.details.get("compensation", 0.0))
                for event in (
                    manager_cancellation_events + pickup_failure_events
                )
            ),
            2,
        ),
        "risk_exposures": int(event_types.get("pickup_risk_exposed", 0)),
        "pickup_risk_attribution": pickup_risk_attribution,
        "fallback_inquiries": int(event_types.get("fallback_inquired", 0)),
        "preauthorized_recovery_plan_candidates_frozen": sum(
            len(commitment.preauthorized_recovery_plans)
            for commitment in state.reservation_commitments
        ),
        "preauthorized_recovery_plan_selections": int(
            event_types.get("preauthorized_recovery_plan_selected", 0)
        ),
        "recovery_price_credits": round(
            sum(
                float(line.recovery_price_credit)
                for commitment in state.reservation_commitments
                for line in commitment.lines
            ),
            2,
        ),
        "recovery_deposit_credits": round(
            sum(
                float(line.recovery_deposit_credit)
                for commitment in state.reservation_commitments
                for line in commitment.lines
            ),
            2,
        ),
        "customer_cancellations": state.cumulative_reservation_customer_cancellations,
        "weather_customer_cancellations": int(
            event_types.get("customer_cancelled_weather", 0)
        ),
        "non_weather_customer_cancellations": int(
            event_types.get("customer_cancelled_non_weather", 0)
        ),
        "weather_reviews_kept_original": int(
            event_types.get("weather_review_no_change", 0)
        ),
        "reschedules": state.cumulative_reservation_reschedules,
        "reschedule_request_tail": reschedule_tail_summary,
        "extensions_requested": state.cumulative_reservation_extensions_requested,
        "extensions_approved": state.cumulative_reservation_extensions_approved,
        "late_returns": state.cumulative_reservation_late_returns,
        "prepayments_collected": state.cumulative_reservation_prepayments,
        "prepayments_refunded": state.cumulative_reservation_prepayment_refunds,
        "compensation": state.cumulative_reservation_compensation,
        "proactive_contingency": contingency_diagnostics,
        "service_reputation": {
            "enabled": bool(reputation_snapshots),
            "latest_snapshot": (
                reputation_snapshots[-1] if reputation_snapshots else None
            ),
            "evaluable_quoted_requests": reputation_evaluable_requests,
            "estimated_acceptances_suppressed_vs_zero_penalty": int(
                reputation_shadow_flips
            ),
            "estimated_acceptance_rate_effect_percentage_points": (
                round(
                    -100.0
                    * reputation_shadow_flips
                    / reputation_evaluable_requests,
                    4,
                )
                if reputation_evaluable_requests
                else None
            ),
            "counterfactual_scope": (
                "request-local score replay; not a dynamic policy counterfactual"
            ),
        },
        "optional_line_credits": state.cumulative_optional_line_credits,
        "unearned_revenue_liability_end": state.unearned_revenue_liability,
        **(
            {
                "rent_cash_audit": {
                    "commitments": len(cash_audit_rows),
                    "prepayment_collected": round(
                        sum(
                            float(row.get("prepayment_collected", 0.0))
                            for row in cash_audit_rows
                        ),
                        2,
                    ),
                    "prepayment_refunded": round(
                        sum(
                            float(row.get("prepayment_refunded", 0.0))
                            for row in cash_audit_rows
                        ),
                        2,
                    ),
                    "rent_collected_at_pickup": round(
                        sum(
                            float(row.get("rent_collected_at_pickup", 0.0))
                            for row in cash_audit_rows
                        ),
                        2,
                    ),
                    "post_pickup_rent_collected": round(
                        sum(
                            float(
                                row.get("post_pickup_rent_collected", 0.0)
                            )
                            for row in cash_audit_rows
                        ),
                        2,
                    ),
                    "gross_rent_cash_collected": round(
                        sum(
                            float(row["gross_rent_cash_collected"])
                            for row in cash_audit_rows
                        ),
                        2,
                    ),
                    "net_rent_cash_collected": round(
                        sum(
                            float(row["net_rent_cash_collected"])
                            for row in cash_audit_rows
                        ),
                        2,
                    ),
                }
            }
            if cash_audit_rows
            else {}
        ),
        "selected_or_committed_asset_days": selected_or_committed_asset_days,
        "accepted_asset_days": accepted_asset_days,
        "requested_service_unit_days": sum(
            int(row.get("requested_service_unit_days") or 0) for row in rows
        ),
        "accepted_service_unit_days": sum(
            int(row.get("accepted_service_unit_days") or 0) for row in rows
        ),
        "committed_service_unit_days": sum(
            int(row.get("committed_service_unit_days") or 0) for row in rows
        ),
        "optional_removed_service_unit_days": sum(
            int(row.get("optional_removed_service_unit_days") or 0)
            for row in rows
        ),
        "recovery_omitted_service_unit_days": sum(
            int(row.get("recovery_omitted_service_unit_days") or 0)
            for row in rows
        ),
        "declined_or_expired_service_unit_days": sum(
            int(row.get("declined_or_expired_service_unit_days") or 0)
            for row in rows
        ),
        "accepted_line_authorization_set_size_counts": dict(
            sorted(authorization_sizes.items())
        ),
        "legacy_exact_candidate_set_size_counts": dict(
            sorted(legacy_candidate_sizes.items())
        ),
        "eligible_profile_set_size_counts": dict(
            sorted(eligible_profile_sizes.items())
        ),
        "inquiry_latent_set_size_counts": dict(
            sorted(inquiry_latent_sizes.items())
        ),
        "recovery_set_size_counts": dict(sorted(recovery_sizes.items())),
        "legacy_mixed_set_size_counts": dict(
            sorted(legacy_mixed_sizes.items())
        ),
        "latent_snapshot_provenance_counts": dict(
            sorted(snapshot_provenance.items())
        ),
        "fulfillment_mode_counts": dict(sorted(fulfillment_modes.items())),
        "reservation_contract_mode_counts": dict(sorted(contract_modes.items())),
        "profile_selection_counts": dict(sorted(profile_selections.items())),
        "profile_family_selection_counts": dict(sorted(profile_families.items())),
        "profile_deposit_policy_counts": dict(sorted(deposit_policies.items())),
        "pairwise_quote_requests": sum(
            row.get("quote_action") == "quote" for row in pairwise_requests
        ),
        "pairwise_aggregate_customer_rejections": sum(
            row.get("request_status") == "customer_declined_quote"
            for row in pairwise_requests
        ),
        "pairwise_menu_size_counts": dict(sorted(pairwise_menu_sizes.items())),
        "pairwise_offer_exposure_counts": dict(sorted(pairwise_exposures.items())),
        "pairwise_selected_offer_counts": dict(sorted(pairwise_selections.items())),
        "pairwise_selection_position_counts": dict(
            sorted(pairwise_selection_positions.items())
        ),
        "pairwise_price_spread": {
            "count": len(pairwise_price_spreads),
            "median": (
                round(statistics.median(pairwise_price_spreads), 2)
                if pairwise_price_spreads
                else None
            ),
            "maximum": (
                max(pairwise_price_spreads) if pairwise_price_spreads else None
            ),
        },
        "complete_contract_quote_requests": sum(
            request.quote_action == "quote"
            for request in complete_contract_requests
        ),
        "complete_contract_option_count_counts": dict(
            sorted(complete_contract_option_counts.items())
        ),
        "complete_contract_selected_position_counts": dict(
            sorted(complete_contract_selected_positions.items())
        ),
        "complete_contract_offer_exposure_counts": dict(
            sorted(complete_contract_offer_exposures.items())
        ),
        "complete_contract_optional_disposition_counts": dict(
            sorted(complete_contract_optional_dispositions.items())
        ),
        "complete_contract_feedback_response_requests": sum(
            request.feedback_response_available is True
            for request in complete_contract_requests
        ),
        "complete_contract_assessment_category_counts": dict(
            sorted(assessment_categories.items())
        ),
        "complete_contract_assessment_outcome_counts": dict(
            sorted(assessment_outcomes.items())
        ),
        "complete_contract_feedback_latent_topic_counts": dict(
            sorted(feedback_latent_topics.items())
        ),
        "complete_contract_feedback_alignment_by_reported_category": (
            feedback_alignment_by_category
        ),
        "factorized_plan_quote_requests": sum(
            request.quote_action == "quote" for request in factorized_requests
        ),
        "factorized_internal_plan_count_counts": dict(
            sorted(factorized_internal_plan_counts.items())
        ),
        "factorized_customer_authorized_plan_count_counts": dict(
            sorted(factorized_authorized_plan_counts.items())
        ),
        "factorized_confirmation_origin_counts": dict(
            sorted(factorized_confirmation_origins.items())
        ),
        "factorized_confirmation_preference_counts": dict(
            sorted(factorized_confirmation_preferences.items())
        ),
        "factorized_nonpreferred_confirmation_count": sum(
            request.manager_confirmed_plan_id is not None
            and request.manager_confirmed_plan_id
            != request.customer_preferred_plan_id
            for request in factorized_requests
        ),
        "factorized_decline_feedback_category_counts": dict(
            sorted(factorized_decline_feedback_categories.items())
        ),
        "factorized_decline_feedback_tag_counts": dict(
            sorted(factorized_decline_feedback_tags.items())
        ),
        "factorized_decline_feedback_concern_set_counts": dict(
            sorted(factorized_decline_feedback_concern_sets.items())
        ),
        "factorized_decline_feedback_available_requests": sum(
            request.reported_decline_feedback.get("available") is True
            for request in factorized_requests
        ),
        "factorized_customer_line_mention_requests": len(
            factorized_line_mentions
        ),
        "factorized_customer_line_mention_service_counts": dict(
            sorted(factorized_line_mention_service_counts.items())
        ),
        "factorized_customer_line_mention_scope_violations": (
            factorized_line_mention_scope_violations
        ),
        "factorized_equipment_feedback_without_line_mention_requests": sum(
            request.reported_decline_feedback.get("available") is True
            and (
                request.reported_decline_feedback.get("category")
                == EQUIPMENT_FIT_CONCERN
                or EQUIPMENT_CONFIGURATION_TAG
                in request.reported_decline_feedback.get(
                    "reported_concerns", []
                )
            )
            and request.reported_decline_feedback.get(
                "customer_mentioned_line_id"
            )
            is None
            for request in factorized_requests
        ),
        "factorized_negative_feedback_with_authorized_plan_violations": sum(
            bool(request.customer_authorized_plans)
            and bool(request.reported_decline_feedback)
            for request in factorized_requests
        ),
        "factorized_feedback_alignment_by_reported_category": (
            factorized_feedback_alignment
        ),
        "factorized_tag_action_alignment": factorized_tag_action_alignment,
        "factorized_multi_authorized_plan_requests": len(
            factorized_multi_plan_requests
        ),
        "factorized_multi_plan_asset_line_diversity": {
            "count": len(factorized_max_pairwise_line_differences),
            "mean_maximum_different_lines": (
                round(
                    sum(factorized_max_pairwise_line_differences)
                    / len(factorized_max_pairwise_line_differences),
                    6,
                )
                if factorized_max_pairwise_line_differences
                else None
            ),
            "requests_with_at_least_two_different_lines": sum(
                value >= 2 for value in factorized_max_pairwise_line_differences
            ),
        },
        "unaccepted_pairwise_submitted_option_asset_day_bounds": {
            "line_count": len(unaccepted_pairwise_quoted_lines),
            "minimum_total": sum(
                int(row["submitted_option_asset_days_min"])
                for row in unaccepted_pairwise_quoted_lines
            ),
            "maximum_total": sum(
                int(row["submitted_option_asset_days_max"])
                for row in unaccepted_pairwise_quoted_lines
            ),
            "scope": (
                "submitted options on quoted lines in requests without a "
                "commitment; descriptive bounds, not selected asset demand"
            ),
        },
        "staging_actions": state.cumulative_reservation_staging_actions,
        "staged_authorized_fallbacks": (
            state.cumulative_reservation_staged_fallbacks
        ),
        "booking_authorized_substitution_rescues": fulfillment_modes.get(
            "booking_authorized_substitute_fulfilled", 0
        ),
        "inquired_fallback_fulfillments": fulfillment_modes.get(
            "inquired_fallback_fulfilled", 0
        ),
        "opportunity_cost_note": (
            "selected/committed asset-day counts require a determined asset "
            "quantity and are not a cross-funnel demand measure; service-unit-days "
            "are the comparable request/acceptance measure; neither is a causal "
            "counterfactual profit estimate"
        ),
        "service_unit_day_outcome_scope": (
            "accepted and declined_or_expired are request-level outcomes based "
            "on commitment presence and sum to requested; committed excludes "
            "optional lines removed when an otherwise accepted request was quoted"
        ),
    }


def _maintenance_rows(state: WorldState) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in state.rental_activity
        if row.get("event_type")
        in {
            "maintenance_started",
            "maintenance_completed",
            "maintenance_hold_started",
        }
    ]


def _business_value_for_manifest(
    environment: OutfitterEnvironment,
    state: WorldState,
    manifest: dict[str, Any],
) -> dict[str, float]:
    economics = manifest.get("environment_rules", {}).get("economics", {})
    environment_protocol = manifest.get("controller", {}).get(
        "environment_protocol"
    )
    if (
        environment_protocol
        in {
            "outfitter-seasonal-v3",
            "outfitter-seasonal-v4",
            "outfitter-seasonal-v5",
            "outfitter-seasonal-v6",
            "outfitter-seasonal-v7-reservations",
            "outfitter-seasonal-v8-disjunctive-reservations",
            "outfitter-seasonal-v9-contract-quotes",
            "outfitter-seasonal-v10-explicit-contract-quotes",
            "outfitter-seasonal-v11-coherent-bundle-latents",
            "outfitter-seasonal-v12-compact-ids-pagination",
            "outfitter-seasonal-v13-lifecycle-closure",
            "outfitter-seasonal-v14-lifecycle-cash-salience",
            "outfitter-seasonal-v15-lifecycle-economic-balance",
            "outfitter-seasonal-v16-reservation-promise-profiles",
        }
        or "retail_liquidation_market" in economics
    ):
        return environment.business_value(state)
    if "secondhand_market" in economics:
        values = environment.business_value(state)
        retail_book = sum(
            float(row["quantity"]) * float(row["avg_unit_cost"])
            for row in state.retail_inventory.values()
        )
        legacy_retail = round(
            float(economics.get("terminal_retail_salvage_rate", 0.50))
            * retail_book,
            2,
        )
        retail_delta = legacy_retail - float(values["retail_inventory_value"])
        legacy_in_transit = sum(
            float(item["line_cost"])
            * (
                0.55
                if environment.offers[str(item["offer_id"])].asset_kind
                == "durable"
                else float(economics.get("terminal_retail_salvage_rate", 0.50))
            )
            for order in state.orders
            if order.status == "in_transit"
            for item in order.items
        )
        transit_delta = legacy_in_transit - float(
            values["in_transit_going_concern_value"]
        )
        values["retail_inventory_value"] = legacy_retail
        values["in_transit_going_concern_value"] = round(
            legacy_in_transit, 2
        )
        for key in (
            "official_liquidation_value",
            "liquidation_business_value",
            "business_value",
            "efficiency_adjusted_value",
        ):
            values[key] = round(float(values[key]) + retail_delta, 2)
        for key in (
            "contract_adjusted_economic_value",
            "going_concern_business_value",
        ):
            values[key] = round(
                float(values[key]) + retail_delta + transit_delta, 2
            )
        return values
    retail_book = sum(
        float(row["quantity"]) * float(row["avg_unit_cost"])
        for row in state.retail_inventory.values()
    )
    rental_remaining = sum(
        asset.unit_cost * environment._remaining_life_ratio(asset)
        for asset in state.assets
        if asset.ownership_type == "owned" and asset.status != "retired"
    )
    retail_rate = float(economics.get("terminal_retail_salvage_rate", 0.50))
    rental_rate = float(
        economics.get("terminal_rental_remaining_life_salvage_rate", 0.60)
    )
    repair_liability = sum(
        environment.repair_cost(asset)
        for asset in state.assets
        if asset.status == "damaged"
    )
    retail_value = retail_rate * retail_book
    rental_value = rental_rate * rental_remaining
    in_transit_book = sum(
        float(order.total_cost)
        for order in state.orders
        if order.status == "in_transit" and order.procurement_mode == "owned"
    )
    value = (
        state.cash
        + retail_value
        + rental_value
        - state.deposit_liability
        - repair_liability
    )
    book_net_assets = (
        state.cash
        - state.deposit_liability
        + retail_book
        + rental_remaining
        + in_transit_book
    )
    return {
        "cash": round(state.cash, 2),
        "in_transit_value": 0.0,
        "in_transit_book_value": round(in_transit_book, 2),
        "retail_inventory_book_value": round(retail_book, 2),
        "retail_inventory_value": round(retail_value, 2),
        "rental_remaining_life_book_value": round(rental_remaining, 2),
        "rental_remaining_life_value": round(rental_value, 2),
        "deposit_liability": round(state.deposit_liability, 2),
        "damaged_repair_liability": round(repair_liability, 2),
        "book_net_assets": round(book_net_assets, 2),
        "going_concern_business_value": round(value, 2),
        "liquidation_business_value": round(value, 2),
        "business_value": round(value, 2),
    }


def _tool_usage(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(row["name"] for row in tools)
    errors = Counter(row["name"] for row in tools if not row["ok"])
    agent_calls = Counter(
        row["name"] for row in tools if row.get("origin") == "agent"
    )
    automatic_calls = Counter(
        row["name"] for row in tools if row.get("origin") == "automatic"
    )
    agent_errors = Counter(
        row["name"]
        for row in tools
        if row.get("origin") == "agent" and not row["ok"]
    )
    automatic_errors = Counter(
        row["name"]
        for row in tools
        if row.get("origin") == "automatic" and not row["ok"]
    )
    generation_truncation_errors = Counter(
        row["name"]
        for row in tools
        if row.get("generation_truncation") and not row["ok"]
    )
    result_sizes: defaultdict[str, list[int]] = defaultdict(list)
    for row in tools:
        result_sizes[row["name"]].append(int(row["result_characters"]))
    return [
        {
            "tool": name,
            "calls": counts[name],
            "errors": errors[name],
            "agent_calls": agent_calls[name],
            "automatic_calls": automatic_calls[name],
            "unknown_origin_calls": counts[name] - agent_calls[name] - automatic_calls[name],
            "agent_errors": agent_errors[name],
            "automatic_errors": automatic_errors[name],
            "unknown_origin_errors": errors[name] - agent_errors[name] - automatic_errors[name],
            "generation_truncation_errors": generation_truncation_errors[name],
            "business_errors": (
                errors[name] - generation_truncation_errors[name]
            ),
            "result_characters_total": sum(result_sizes[name]),
            "result_characters_average": round(
                sum(result_sizes[name]) / len(result_sizes[name]), 1
            ),
            "result_characters_maximum": max(result_sizes[name]),
        }
        for name in sorted(counts, key=lambda value: (-counts[value], value))
    ]


def _quote_history_query_audit(
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    calls = [
        row
        for row in tools
        if row["name"] == "view_reservation"
        and row.get("arguments", {}).get("view") == "quote_history"
    ]
    query_types: Counter[str] = Counter()
    successful_types: Counter[str] = Counter()
    error_types: Counter[str] = Counter()
    filter_dimensions: Counter[str] = Counter()
    sort_orders: Counter[str] = Counter()
    result_sizes_by_type: defaultdict[str, list[int]] = defaultdict(list)
    filter_keys = (
        "offer_ids",
        "service_keys",
        "quote_outcomes",
        "menu_sizes",
    )
    for row in calls:
        arguments = dict(row.get("arguments") or {})
        dimensions: list[str] = []
        if arguments.get("identifiers"):
            dimensions.append("identifiers")
        if "from_day" in arguments or "to_day" in arguments:
            dimensions.append("created_day")
        dimensions.extend(
            key for key in filter_keys if arguments.get(key)
        )
        if not dimensions:
            query_type = "full"
        elif dimensions == ["identifiers"]:
            query_type = "exact"
        else:
            query_type = "filtered"
        query_types[query_type] += 1
        (successful_types if row["ok"] else error_types)[query_type] += 1
        filter_dimensions.update(dimensions)
        sort_orders[str(arguments.get("sort_order") or "default")] += 1
        result_sizes_by_type[query_type].append(
            int(row.get("result_characters", 0))
        )

    def size_summary(values: list[int]) -> dict[str, int | float]:
        return {
            "count": len(values),
            "total": sum(values),
            "median": round(float(statistics.median(values)), 1) if values else 0.0,
            "maximum": max(values, default=0),
        }

    all_sizes = [int(row.get("result_characters", 0)) for row in calls]
    return {
        "calls": len(calls),
        "agent_calls": sum(row.get("origin") == "agent" for row in calls),
        "automatic_calls": sum(
            row.get("origin") == "automatic" for row in calls
        ),
        "successful_calls": sum(row["ok"] for row in calls),
        "error_calls": sum(not row["ok"] for row in calls),
        "query_type_counts": dict(sorted(query_types.items())),
        "successful_query_type_counts": dict(sorted(successful_types.items())),
        "error_query_type_counts": dict(sorted(error_types.items())),
        "filter_dimension_counts": dict(sorted(filter_dimensions.items())),
        "sort_order_counts": dict(sorted(sort_orders.items())),
        "result_characters": size_summary(all_sizes),
        "result_characters_by_query_type": {
            query_type: size_summary(result_sizes_by_type[query_type])
            for query_type in ("full", "exact", "filtered")
        },
    }


def _memory_usage(
    tools: list[dict[str, Any]], state: WorldState
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    names = {
        "memory_list",
        "memory_search",
        "memory_read",
        "memory_write",
        "memory_delete",
    }
    memory_calls = [row for row in tools if row["name"] in names]
    call_counts = Counter(row["name"] for row in memory_calls)
    successful_counts = Counter(row["name"] for row in memory_calls if row["ok"])
    failed_counts = Counter(row["name"] for row in memory_calls if not row["ok"])
    per_document: dict[str, dict[str, Any]] = {}
    for call in memory_calls:
        if call["name"] in {"memory_list", "memory_search"}:
            continue
        raw_name = call["arguments"].get("name")
        if raw_name is None:
            continue
        name = str(raw_name).lower()
        row = per_document.setdefault(
            name,
            {
                "name": name,
                "first_write_day": None,
                "last_write_day": None,
                "writes": 0,
                "successful_reads": 0,
                "failed_reads": 0,
                "deletes": 0,
            },
        )
        if call["name"] == "memory_write" and call["ok"]:
            row["writes"] += 1
            row["first_write_day"] = (
                call["day"]
                if row["first_write_day"] is None
                else min(row["first_write_day"], call["day"])
            )
            row["last_write_day"] = (
                call["day"]
                if row["last_write_day"] is None
                else max(row["last_write_day"], call["day"])
            )
        elif call["name"] == "memory_read":
            key = "successful_reads" if call["ok"] else "failed_reads"
            row[key] += 1
        elif call["name"] == "memory_delete" and call["ok"]:
            row["deletes"] += 1

    final_documents = state.memory_documents
    rows: list[dict[str, Any]] = []
    for name in sorted(set(per_document) | set(final_documents)):
        activity = per_document.get(
            name,
            {
                "name": name,
                "first_write_day": None,
                "last_write_day": None,
                "writes": 0,
                "successful_reads": 0,
                "failed_reads": 0,
                "deletes": 0,
            },
        )
        document = final_documents.get(name)
        rows.append(
            {
                "name": name,
                "created_day": (
                    document.created_day if document is not None else activity["first_write_day"]
                ),
                "updated_day": (
                    document.updated_day if document is not None else activity["last_write_day"]
                ),
                "writes": activity["writes"],
                "successful_reads": activity["successful_reads"],
                "failed_reads": activity["failed_reads"],
                "deletes": activity["deletes"],
                "characters": len(document.content) if document is not None else 0,
                "still_exists": document is not None,
            }
        )
    written = [row for row in rows if row["writes"] > 0]
    final_characters = sum(len(row.content) for row in final_documents.values())
    search_calls = [row for row in memory_calls if row["name"] == "memory_search"]
    summary = {
        "calls": {name: call_counts[name] for name in sorted(names)},
        "successful_calls": {name: successful_counts[name] for name in sorted(names)},
        "failed_calls": {name: failed_counts[name] for name in sorted(names)},
        "distinct_documents_written": len(written),
        "final_documents": len(final_documents),
        "final_total_characters": final_characters,
        "final_average_characters": (
            round(final_characters / len(final_documents), 2) if final_documents else 0.0
        ),
        "written_exactly_once": sum(row["writes"] == 1 for row in written),
        "never_read": sum(row["successful_reads"] == 0 for row in written),
        "read_exactly_once": sum(row["successful_reads"] == 1 for row in written),
        "read_exactly_twice": sum(row["successful_reads"] == 2 for row in written),
        "read_more_than_twice": sum(row["successful_reads"] > 2 for row in written),
        "successful_memory_searches": sum(row["ok"] for row in search_calls),
        "failed_memory_searches": sum(not row["ok"] for row in search_calls),
        "zero_result_memory_searches": sum(
            row["ok"] and int(row["data"].get("matched_documents", 0)) == 0
            for row in search_calls
        ),
        "maximum_memory_list_result_characters": max(
            (
                row["result_characters"]
                for row in memory_calls
                if row["name"] == "memory_list"
            ),
            default=0,
        ),
        "maximum_memory_search_result_characters": max(
            (row["result_characters"] for row in search_calls),
            default=0,
        ),
    }
    return summary, rows


def _markdown(report: dict[str, Any]) -> str:
    headline = report["headline"]
    finance = report["financials"]
    demand = report["demand"]
    reservations = report.get("reservations", {})
    protocol = report.get("reservation_protocol_audit", {})
    memory = report["memory_usage"]
    quote_history_audit = report.get("quote_history_query_audit", {})
    quote_workspace_audit = report.get("quote_workspace_audit", {})
    evaluation_score = headline["official_liquidation_value"]
    score_text = "n/a" if evaluation_score is None else f"{evaluation_score:.2f}"
    final_value = headline["official_liquidation_value"]
    value_text = "n/a" if final_value is None else f"{final_value:.2f}"
    horizon_text = (
        f"; decision days completed: {headline['completed_decision_days']}"
        f"/{headline['decision_horizon_day']}"
        f"; runoff days completed: {headline['completed_runoff_days']}"
        f"; final scoring day: {headline['runoff_end_day']}"
        if headline.get("decision_horizon_day") is not None
        and headline.get("runoff_end_day") is not None
        else ""
    )
    return "\n".join(
        [
            "# OutfitterBench run report",
            "",
            f"- Run: `{report['run']['run_id']}`",
            f"- Status: `{headline['status']}`; completed calendar days: {headline['completed_days']}{horizon_text}",
            f"- Liquidation policy: {headline.get('liquidation_policy', 'legacy')}",
            f"- Official liquidation value (model-cost independent): {score_text}",
            f"- Contract-adjusted economic value at stop: {headline['contract_adjusted_economic_value_at_stop']:.2f}",
            f"- Efficiency-adjusted value (standardized token charge): {headline['efficiency_adjusted_value_at_stop']:.2f}",
            f"- Official-value change at stop: {headline['official_liquidation_value_change']:.2f} ({headline['official_liquidation_value_change_pct']:.1f}%)",
            f"- Legacy execution liquidation value: {value_text}",
            f"- Book net assets at stop: {headline['book_net_assets_at_stop']:.2f}",
            f"- Operating P&L: {finance['operating_pnl']:.2f}; after retail liquidation losses: {finance['business_pnl_after_retail_liquidation']:.2f}; revenue incl. damage deductions: {finance['operating_revenue']:.2f}; rental-asset sale proceeds: {finance['asset_sale_proceeds']:.2f}; retail liquidation proceeds: {finance['retail_liquidation_proceeds']:.2f}",
            f"- Requested / fulfilled units: {demand['requested_units']} / {demand['fulfilled_units']}",
            f"- Horizon-eligible requested units / fulfillment rate: {demand['eligible_requested_units']} / {demand['eligible_fulfillment_rate']}",
            f"- Lost demand: outside {demand['outside_rejections']}, stockout {demand['stockout_rejections']}, service closed {demand['service_closed_rejections']}, capacity policy {demand.get('capacity_policy_rejections', 0)}, horizon {demand['horizon_rejections']}",
            f"- Baskets: full {demand['full_baskets']}, partial {demand['partial_baskets']}, unfilled {demand['unfilled_baskets']}",
            f"- Reservations: inquiries {reservations.get('requests', 0)}, decisions {reservations.get('decision_counts', {})}, customer quote declines {reservations.get('customer_quote_declines', 0)}, confirmed {reservations.get('confirmed', 0)}, pickups {reservations.get('pickups', 0)}, pickup failures {reservations.get('pickup_failures', 0)} (after decision horizon {reservations.get('pickup_failures_after_decision_horizon', 0)}), manager risk cancellations {reservations.get('manager_cancellations', 0)}, customer cancellations {reservations.get('customer_cancellations', 0)}, compensation {reservations.get('compensation', 0.0):.2f}",
            f"- Reschedule-request tail: {reservations.get('reschedule_request_tail', {})}",
            *([f"- V32 typed reservation counts and rent cash (not allocated profit): {reservations['service_commitments']}"]
              if reservations.get("service_commitments") else []),
            f"- Reservation recovery: fallback inquiries {reservations.get('fallback_inquiries', 0)}, dormant complete-plan candidates {reservations.get('preauthorized_recovery_plan_candidates_frozen', 0)}, explicit complete-plan selections {reservations.get('preauthorized_recovery_plan_selections', 0)}, price/deposit protection credits {reservations.get('recovery_price_credits', 0.0):.2f}/{reservations.get('recovery_deposit_credits', 0.0):.2f}",
            "- V38 proactive contingency (diagnostic, not an independent reward): "
            f"{{{', '.join(f'{key}={value}' for key, value in reservations.get('proactive_contingency', {}).items() if key != 'daily')}}}",
            f"- Pickup-risk attribution: {reservations.get('pickup_risk_attribution', {})}. This describes the physical gap at exposure; preceding changes are reported separately and are not asserted to have caused the risk.",
            f"- Reservation demand units: requested/accepted-request/declined-or-expired-request service-unit-days {reservations.get('requested_service_unit_days', 0)}/{reservations.get('accepted_service_unit_days', 0)}/{reservations.get('declined_or_expired_service_unit_days', 0)}; committed/optional-removed service-unit-days {reservations.get('committed_service_unit_days', 0)}/{reservations.get('optional_removed_service_unit_days', 0)}; selected-or-committed/accepted asset-days {reservations.get('selected_or_committed_asset_days', 0)}/{reservations.get('accepted_asset_days', 0)}",
            f"- Reservation protocol audit: opening mismatches {protocol.get('opening_inbox_mismatch_count', 0)}/{protocol.get('opening_reports_audited', 0)}, complete explicit quotes {protocol.get('complete_explicit_quotes', 0)}/{protocol.get('explicit_quotes_audited', 0)}, forbidden public-term hits {protocol.get('forbidden_public_term_hit_total', 0)}",
            f"- Factorized decline feedback: legacy categories {reservations.get('factorized_decline_feedback_category_counts', {})}; V23 tags {reservations.get('factorized_decline_feedback_tag_counts', {})}; concern sets {reservations.get('factorized_decline_feedback_concern_set_counts', {})}; tag action alignment {reservations.get('factorized_tag_action_alignment', {})}",
            f"- Quote-history queries: {quote_history_audit.get('calls', 0)} total, types {quote_history_audit.get('query_type_counts', {})}, errors {quote_history_audit.get('error_calls', 0)}, result characters {quote_history_audit.get('result_characters', {})}",
            f"- Quote workspaces: {quote_workspace_audit.get('calls', 0)} total, errors {quote_workspace_audit.get('error_calls', 0)}, projections {quote_workspace_audit.get('projection_counts', {})}, returned requests {quote_workspace_audit.get('returned_request_records', 0)} ({quote_workspace_audit.get('distinct_request_ids', 0)} distinct), result characters {quote_workspace_audit.get('result_characters', {})}; request-set reuse {quote_workspace_audit.get('request_set_reuse', {})}; legacy line-option calls {quote_workspace_audit.get('legacy_option_calls', 0)}",
            f"- Bundle latent/snapshot audit: shared-field mismatches {protocol.get('bundle_shared_field_mismatch_counts', {})}, storage/reuse adjustment violations {protocol.get('storage_category_adjustment_violation_count', 0)}/{protocol.get('reuse_category_adjustment_violation_count', 0)}, snapshot invariant violations {len(protocol.get('snapshot_invariant_violations', []))}",
            f"- Operational baskets: full {demand['operational_full_baskets']}, partial {demand['operational_partial_baskets']}, unfilled {demand['operational_unfilled_baskets']}",
            f"- Model calls: {report['agent_usage']['model_calls']}; agent tool calls: {report['agent_usage']['agent_tool_calls']}; automatic tool calls: {report['agent_usage']['automatic_tool_calls']}; unknown origin: {report['agent_usage'].get('unknown_origin_tool_calls', 0)}",
            f"- Model/execution identity: {report['agent_usage'].get('model_identity_status_counts', {})}; snapshot: {report.get('analysis_snapshot')}",
            f"- Invalid-action episodes: {report['agent_usage']['invalid_action_episodes']}; failed agent call records: {report['agent_usage']['agent_tool_errors']}; read/state/terminal business errors: {report['agent_usage']['business_tool_errors_by_effect']}; rejected-batch call records: {report['agent_usage']['rejected_batched_tool_calls']}",
            f"- Structured quote-validation failures: {report['agent_usage']['structured_quote_validation_batch_records']} failed batches containing {report['agent_usage']['structured_quote_validation_invalid_items']} invalid items",
            f"- Rental-capacity queries: {report['agent_usage']['rental_capacity_tool_calls']} calls, {report['agent_usage']['rental_capacity_tool_errors']} errors; in-transit actual-delivery privacy violations: {report['agent_usage']['rental_capacity_in_transit_actual_delivery_violations']}",
            f"- Walk-in capacity-limit changes: {report['agent_usage'].get('walk_in_capacity_limit_tool_calls', 0)} calls, {report['agent_usage'].get('walk_in_capacity_limit_tool_errors', 0)} errors, {report['agent_usage'].get('walk_in_capacity_limit_warning_records', 0)} grandfathered-overlimit warnings",
            f"- Reservation-view result characters by view: {report['agent_usage']['reservation_view_result_characters_by_view']}",
            f"- Generation finish reasons: {report['agent_usage']['finish_reason_counts']}; length truncations: {report['agent_usage']['generation_truncations']}; truncated tool JSON: {report['agent_usage']['generation_truncated_tool_json']}",
            f"- Active context message characters (min/median/max): {report['agent_usage']['context_message_characters']}; sliding-window requests: {report['agent_usage']['sliding_window_requests']}; maximum omitted messages: {report['agent_usage']['maximum_sliding_window_omitted_messages']}",
            f"- Tool-result characters: {report['agent_usage']['tool_result_characters']}; largest result: {report['agent_usage']['maximum_tool_result']}",
            f"- Same-result read-only reuse: {report.get('read_only_result_reuse_audit', {})}",
            f"- Read-only batches: {report['agent_usage']['read_only_tool_batches']}; rejected batches: {report['agent_usage']['rejected_tool_batches']}",
            f"- Memory documents: {memory['final_documents']}; total characters: {memory['final_total_characters']}; never read: {memory['never_read']}; read once: {memory['read_exactly_once']}",
            f"- Memory searches: {memory['successful_memory_searches']} successful, {memory['failed_memory_searches']} failed, {memory['zero_result_memory_searches']} with no matches",
            "- Purchased-rental contribution concentration flags (>40% of positive "
            f"cohort contribution): {report['review_flags']['purchased_rental_offers_above_40pct_positive_contribution']}",
            f"- Database integrity: `{report['database_integrity']}`",
            "",
            "## Figures",
            "",
            *[f"- `{name}`" for name in PLOT_FILES if name in report["files"]],
            "",
            "## Tables",
            "",
            "- `daily_metrics.csv`",
            "- `service_daily_metrics.csv` and `service_summary.csv`",
            "- `product_fulfillment_daily_metrics.csv` and `product_fulfillment_summary.csv`",
            "- `offer_fulfillment_daily_metrics.csv` and `offer_fulfillment_summary.csv`",
            "- `product_weather_contribution.csv`, `offer_concentration.csv`, and `service_concentration.csv`",
            "- `rental_activity.csv` and `rental_activity_summary.csv`",
            "- `purchased_rental_contribution_by_offer.csv` and `purchased_rental_contribution_by_product.csv`",
            "- `counter_interactions.csv` and `demand_cube.csv`",
            "- `orders.csv`, `rental_contracts.csv`, `asset_sales.csv`, and `retail_liquidations.csv`",
            "- `reservation_requests.csv`, `reservation_lines.csv`, `reservation_events.csv`, and `maintenance_events.csv`",
            "- `final_retail_inventory.csv` and `final_rental_assets.csv`",
            "- `tool_usage.csv`",
            "- `memory_documents.csv`",
            "",
            "`daily_operating_pnl` excludes purchases, deposits/refunds, dealer proceeds, retail-liquidation losses, and model fees; it is therefore not expected to equal official value change. A newly purchased durable immediately moves from supplier cost to its public secondhand residual basis. `daily_business_pnl_after_liquidation` separately subtracts realized retail-liquidation losses. Contract-adjusted/going-concern value is a close-now value for current obligations, not a capitalization of unknown future demand. Gross-contribution leave-one-out figures and purchased-rental cohort contribution are descriptive, not causal demand counterfactuals; purchased-asset holding cost is an approximate rule-based allocation. Legacy fields retain the execution-time liquidation score; explicit official values add back model charges from legacy runs and apply a common token-price efficiency adjustment.",
            "",
        ]
    )












def _counter_interaction_rows(state: WorldState) -> list[dict[str, Any]]:
    rows = []
    for report in state.daily_history:
        for interaction in report.get("counter_interactions", []):
            request = interaction["request"]
            availability = interaction["preferred_availability"]
            response = interaction["response"]
            quotes = interaction.get("quote_lines", [])
            rows.append(
                {
                    "interaction_id": interaction["interaction_id"],
                    "day": interaction["day"],
                    "day_type": interaction["day_type"],
                    "weather": interaction["weather"],
                    "visible_party_context": interaction["visible_party_context"],
                    **request,
                    "preferred_group_state": availability["state"],
                    "preferred_available_units": availability["available_units"],
                    "accepted_units": response["accepted_units"],
                    "unfulfilled_units": response["unfulfilled_units"],
                    "disposition": response["disposition"],
                    "quoted_offer_ids": "|".join(
                        str(line.get("offer_id")) for line in quotes
                    ),
                    "quote_kinds": "|".join(
                        str(line.get("quote_kind")) for line in quotes
                    ),
                    "quoted_revenue": round(
                        sum(float(line.get("revenue", 0.0)) for line in quotes), 2
                    ),
                }
            )
    return rows


def _demand_cube_rows(state: WorldState) -> list[dict[str, Any]]:
    return [
        dict(row)
        for report in state.daily_history
        for row in report.get("demand_cube", [])
    ]



def analyze_run(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    plots: bool = True,
    dpi: int = 160,
) -> dict[str, Any]:
    """Analyze one consistent committed SQLite snapshot, including WAL rows."""
    original = Path(run_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve() if output_dir else original/'analysis'
    with tempfile.TemporaryDirectory(prefix='outfitter-analysis-') as scratch:
        snapshot = Path(scratch)
        manifest = json.loads((original/'manifest.json').read_text())
        for name in ('manifest.json','summary.json','system_prompt.md','business_rules.json'):
            if (original/name).is_file():shutil.copy2(original/name,snapshot/name)
        data_name=manifest.get('runtime_data','data')
        # Frozen runtime data is immutable; only the transactional DB needs a copy.
        (snapshot/'data').symlink_to((original/data_name).resolve(),target_is_directory=True)
        manifest['runtime_data']='data'
        (snapshot/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False))
        with sqlite3.connect(f'file:{original / "run.sqlite3"}?mode=ro',uri=True) as source:
            with sqlite3.connect(snapshot/'run.sqlite3') as target:
                source.backup(target)
                state_raw=target.execute('select value from state where id=1').fetchone()[0]
                stamp={'captured_at_utc':datetime.now(timezone.utc).isoformat(),
                    'state_sha256':hashlib.sha256(state_raw.encode()).hexdigest(),
                    'max_message_seq':target.execute('select max(seq) from messages').fetchone()[0],
                    'max_model_call_id':target.execute('select max(id) from model_calls').fetchone()[0],
                    'tool_call_count':target.execute('select count(*) from tool_calls').fetchone()[0],
                    'state_day':json.loads(state_raw)['day']}
        return _analyze_snapshot(snapshot, output_dir=destination, plots=plots, dpi=dpi,
                                 original_run_dir=original, snapshot_metadata=stamp)


def _analyze_snapshot(run_dir, *, output_dir=None, plots=True, dpi=160,
                      original_run_dir=None, snapshot_metadata=None):
    if not 72 <= dpi <= 600:
        raise ValueError("dpi must be in [72, 600]")
    path, manifest, summary, config, dataset, environment, state = _load_run(run_dir)
    tool_call_protocol = manifest.get("controller", {}).get(
        "tool_call_protocol", "first_only"
    )
    models, tools, integrity = _load_events(
        path, tool_call_protocol=tool_call_protocol
    )
    daily = _daily_rows(state, models, tools)
    standardized_model_cost, standardized_input_rate, standardized_output_rate = (
        _standardized_model_cost(models, config, manifest)
    )
    model_cost_charged_to_cash = _model_cost_was_charged_to_cash(manifest)
    models_by_day: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for model in models:
        models_by_day[int(model["day"])].append(model)
    cumulative_standardized_cost = 0.0
    cumulative_execution_model_cost = 0.0
    for row in daily:
        day_models = models_by_day[int(row["day"])]
        daily_standardized_cost = (
            sum(value["input_tokens"] for value in day_models) * standardized_input_rate
            + sum(value["output_tokens"] for value in day_models) * standardized_output_rate
        ) / 1_000_000
        cumulative_standardized_cost += daily_standardized_cost
        cumulative_execution_model_cost += float(row.get("model_cost", 0.0))
        model_cash_adjustment = (
            cumulative_execution_model_cost if model_cost_charged_to_cash else 0.0
        )
        official_liquidation = float(row["liquidation_business_value_end"]) + model_cash_adjustment
        contract_value = float(row["operating_business_value_end"]) + model_cash_adjustment
        row["standardized_model_cost"] = round(daily_standardized_cost, 6)
        row["cumulative_standardized_model_cost"] = round(cumulative_standardized_cost, 6)
        row["official_liquidation_value_end"] = round(official_liquidation, 2)
        row["contract_adjusted_economic_value_end"] = round(contract_value, 2)
        row["efficiency_adjusted_value_end"] = round(
            official_liquidation - cumulative_standardized_cost, 2
        )
    has_complete_going_concern_history = bool(daily) and all(
        row["going_concern_business_value_end"] is not None for row in daily
    )
    if not has_complete_going_concern_history:
        for row in daily:
            row["operating_business_value_end"] = row["book_net_assets_end"]
            row["operating_value_basis"] = "reconstructed_book_net_assets"
            model_cash_adjustment = (
                sum(
                    float(previous.get("model_cost", 0.0))
                    for previous in daily
                    if int(previous["day"]) <= int(row["day"])
                )
                if model_cost_charged_to_cash
                else 0.0
            )
            row["contract_adjusted_economic_value_end"] = round(
                float(row["book_net_assets_end"]) + model_cash_adjustment, 2
            )
    service_daily = _service_rows(state, environment.public_service_keys())
    services = _service_summary(service_daily)
    product_fulfillment_daily = _product_fulfillment_rows(environment, state)
    product_fulfillment = _product_fulfillment_summary(product_fulfillment_daily)
    offer_fulfillment_daily = _offer_fulfillment_rows(environment, state)
    offer_fulfillment = _offer_fulfillment_summary(offer_fulfillment_daily)
    product_weather_contribution = _product_weather_summary(product_fulfillment_daily)
    operating_pnl = sum(float(row["daily_operating_pnl"]) for row in daily)
    business_pnl_after_retail_liquidation = round(
        operating_pnl - state.cumulative_retail_liquidation_loss, 2
    )
    offer_concentration = _concentration_rows(
        offer_fulfillment_daily, level="offer", operating_pnl=operating_pnl
    )
    service_concentration = _concentration_rows(
        offer_fulfillment_daily, level="service", operating_pnl=operating_pnl
    )
    rental_activity = _rental_activity_rows(environment, state)
    rental_activity_summary = _rental_activity_summary(rental_activity)
    (
        purchased_rental_contribution_by_offer,
        purchased_rental_contribution_by_product,
        concentrated_purchased_rental_offers,
    ) = _purchased_rental_contribution(
        environment,
        state,
        completed_days=len(state.daily_history),
    )
    initial_value_components = _business_value_for_manifest(
        environment, environment.reset(), manifest
    )
    initial_value = initial_value_components["business_value"]
    initial_book_net_assets = initial_value_components.get(
        "book_net_assets", initial_value
    )
    initial_operating_value = (
        initial_value_components.get(
            "going_concern_business_value", initial_book_net_assets
        )
        if has_complete_going_concern_history
        else initial_book_net_assets
    )
    final_value = _business_value_for_manifest(environment, state, manifest)
    business_value_at_stop = final_value["business_value"]
    execution_model_cost = float(getattr(state, "cumulative_model_cost", 0.0))
    model_cash_adjustment = execution_model_cost if model_cost_charged_to_cash else 0.0
    official_liquidation_value_at_stop = round(
        business_value_at_stop + model_cash_adjustment, 2
    )
    operating_value_at_stop = (
        final_value.get(
            "going_concern_business_value",
            final_value.get("book_net_assets", business_value_at_stop),
        )
        if has_complete_going_concern_history
        else final_value.get("book_net_assets", business_value_at_stop)
    )
    contract_adjusted_economic_value_at_stop = round(
        operating_value_at_stop + model_cash_adjustment, 2
    )
    initial_official_liquidation_value = round(initial_value, 2)
    initial_contract_adjusted_economic_value = round(initial_operating_value, 2)
    efficiency_adjusted_value_at_stop = round(
        official_liquidation_value_at_stop - standardized_model_cost, 2
    )
    evaluation_score = (
        (-round(state.bankruptcy_shortfall, 2) if state.bankrupt else business_value_at_stop)
        if state.done
        else None
    )
    final_business_value = (
        business_value_at_stop if state.done and not state.bankrupt else None
    )
    total_cost = (
        state.cumulative_purchase_cost
        + state.cumulative_supplier_trial_handling_fees
        + state.cumulative_supplier_trial_usage_fees
        + state.cumulative_cleaning_cost
        + state.cumulative_repair_cost
        + state.cumulative_maintenance_cost
        + state.cumulative_reservation_compensation
        + state.cumulative_emergency_credit_fees
        + state.cumulative_retail_holding_cost
        + state.cumulative_rental_holding_cost
        + state.cumulative_fees
        + state.cumulative_model_cost
    )
    retail = _retail_rows(environment, state)
    assets = _asset_rows(environment, state)
    supplier_trials = _supplier_trial_rows(environment, state)
    durable_capital_exits = _durable_capital_exit_rows(environment, state)
    asset_sales = _asset_sale_rows(environment, state)
    retail_liquidations = _retail_liquidation_rows(environment, state)
    reservation_requests, reservation_lines, reservation_events = (
        _reservation_analysis_rows(
            state,
            extension_cash_audit=(
                config.reservation_contract_mode
                == "factorized_line_options_plan_set_v1"
                and int(dataset.metadata.get("tool_interface_version", 0)) >= 23
            ),
        )
    )
    reservation_summary = _reservation_summary(
        state,
        reservation_requests,
        reservation_lines,
        environment=environment,
    )
    reservation_protocol_audit = _reservation_protocol_audit(path, state, tools)
    maintenance_events = _maintenance_rows(state)
    tool_usage = _tool_usage(tools)
    quote_history_query_audit = _quote_history_query_audit(tools)
    memory_usage, memory_documents = _memory_usage(tools, state)
    tool_effects = dict(manifest.get("controller", {}).get("tool_effects") or {})
    business_error_tools = [
        row
        for row in tools
        if row.get("origin") == "agent"
        and not row["ok"]
        and not row.get("generation_truncation")
    ]
    business_errors_by_effect = Counter(
        str(tool_effects.get(row["name"], "unknown"))
        for row in business_error_tools
    )
    structured_quote_validation_rows = [
        row
        for row in tools
        if row.get("name") == "quote_reservation_inquiries"
        and row.get("data", {}).get("error")
        == "reservation quote batch contains invalid items"
    ]
    capacity_tool_rows = [
        row for row in tools if row.get("name") == "view_rental_capacity"
    ]
    capacity_limit_tool_rows = [
        row
        for row in tools
        if row.get("name") == "set_walk_in_capacity_limits"
    ]
    quote_workspace_rows = [
        row
        for row in tools
        if row.get("name") == "view_reservation_quote_workspace"
    ]
    successful_quote_workspaces = [
        row for row in quote_workspace_rows if row.get("ok")
    ]
    quote_workspace_characters = sorted(
        int(row.get("result_characters", 0))
        for row in successful_quote_workspaces
    )
    quote_workspace_request_ids = {
        str(request.get("request_id"))
        for row in successful_quote_workspaces
        for request in row.get("data", {}).get("requests", [])
        if request.get("request_id") is not None
    }
    legacy_option_rows = [
        row for row in tools if row.get("name") == "view_reservation_options"
    ]
    quote_workspace_audit = {
        "calls": len(quote_workspace_rows),
        "error_calls": sum(not row.get("ok") for row in quote_workspace_rows),
        "projection_counts": dict(
            sorted(
                Counter(
                    str(row.get("data", {}).get("offer_projection") or "unknown")
                    for row in successful_quote_workspaces
                ).items()
            )
        ),
        "returned_request_records": sum(
            int(row.get("data", {}).get("returned_request_count", 0))
            for row in successful_quote_workspaces
        ),
        "distinct_request_ids": len(quote_workspace_request_ids),
        "pages_with_more": sum(
            bool(row.get("data", {}).get("has_more"))
            for row in successful_quote_workspaces
        ),
        "result_characters": {
            "count": len(quote_workspace_characters),
            "total": sum(quote_workspace_characters),
            "median": (
                quote_workspace_characters[len(quote_workspace_characters) // 2]
                if quote_workspace_characters
                else 0
            ),
            "maximum": max(quote_workspace_characters, default=0),
        },
        "dynamic_boundary_claim_violations": sum(
            row.get("data", {}).get("dynamic_operating_data_included") is not False
            or row.get("data", {}).get(
                "recommendation_or_feasibility_included"
            )
            is not False
            for row in successful_quote_workspaces
        ),
        "legacy_option_calls": len(legacy_option_rows),
        "legacy_option_result_characters": sum(
            int(row.get("result_characters", 0)) for row in legacy_option_rows
        ),
    }
    capacity_delivery_privacy_violations = sum(
        event.get("order_status") == "in_transit"
        and "actual_delivery_day" in event
        for row in capacity_tool_rows
        if row.get("ok")
        for event in row.get("data", {}).get("supply_events", [])
    )
    reservation_view_result_characters: defaultdict[str, list[int]] = (
        defaultdict(list)
    )
    for row in tools:
        if row.get("name") != "view_reservation" or not row.get("ok"):
            continue
        view = str(row.get("data", {}).get("view") or "unknown")
        reservation_view_result_characters[view].append(
            int(row.get("result_characters", 0))
        )
    finish_reason_counts = Counter(
        str(row.get("finish_reason") or "missing") for row in models
    )
    context_character_values = sorted(
        int(row.get("context_message_characters", 0)) for row in models
    )
    omitted_message_values = [
        int(row.get("sliding_window_omitted_messages", 0))
        for row in models
    ]
    largest_tool_result = max(
        tools,
        key=lambda row: int(row.get("result_characters", 0)),
        default=None,
    )
    successful_read_only_rows = [
        row
        for row in tools
        if row["ok"] and tool_effects.get(row["name"]) == "read_only"
    ]
    read_only_result_reuse_audit, _read_only_reuse_by_call = (
        _classify_repeated_read_only_calls(
            successful_read_only_rows,
            models,
            signature=lambda row: (
                int(row["day"]),
                row["name"],
                json.dumps(
                    {"ok": True, **row["data"]},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            ),
        )
    )
    workspace_request_set_reuse, workspace_reuse_by_call = (
        _classify_repeated_read_only_calls(
            successful_quote_workspaces,
            models,
            signature=lambda row: (
                (
                    int(row["day"]),
                    tuple(
                        sorted(
                            str(request["request_id"])
                            for request in row.get("data", {}).get(
                                "requests", []
                            )
                        )
                    ),
                    str(
                        row.get("data", {}).get("offer_projection")
                        or "unknown"
                    ),
                )
                if row.get("data", {}).get("requests")
                else None
            ),
        )
    )
    quote_workspace_audit.update(
        {
            "request_set_reuse": workspace_request_set_reuse,
            "request_records_refetched_after_context_eviction": sum(
                int(row.get("data", {}).get("returned_request_count", 0))
                for row in successful_quote_workspaces
                if workspace_reuse_by_call.get(str(row.get("call_id")))
                == "refetched_after_context_eviction"
            ),
        }
    )
    # Compatibility totals preserve the old same-result count while the new
    # audit states whether each recurrence was actually avoidable.
    duplicate_read_only_result_records = int(
        read_only_result_reuse_audit["same_signature_reoccurrences"]
    )
    duplicate_read_only_result_characters = sum(
        int(read_only_result_reuse_audit[category]["characters"])
        for category in (
            "repeated_while_still_in_context",
            "refetched_after_context_eviction",
            "same_model_batch_repetition",
            "unattributed",
        )
    )
    has_split_rejections = any(
        "service_closed_rejections" in row.get("demand", {})
        or "horizon_rejections" in row.get("demand", {})
        for row in state.daily_history
    )
    service_closed_rejections = (
        state.cumulative_service_closed_rejections
        if has_split_rejections
        else state.cumulative_closed_rejections
    )
    eligible_requested = max(
        0, state.cumulative_requested_units - state.cumulative_horizon_rejections
    )
    operational_requested = (
        state.cumulative_fulfilled_units
        + state.cumulative_stockout_rejections
        + service_closed_rejections
        + int(getattr(state, "cumulative_capacity_policy_rejections", 0))
        + int(getattr(state, "cumulative_unpriced_rejections", 0))
    )
    internal_channel_rows = [
        {"day": int(report["day"]), **dict(row)}
        for report in state.daily_history
        for row in report.get("internal_channel_outcomes", [])
    ]
    internal_basket_intent_rows = [
        {"day": int(report["day"]), **dict(row)}
        for report in state.daily_history
        for row in report.get("internal_basket_intents", [])
    ]
    internal_channel_counts = Counter(
        str(row.get("selected_channel") or "unknown")
        for row in internal_channel_rows
    )
    basket_intent_status_counts = Counter(
        str(row.get("status") or "unknown")
        for row in internal_basket_intent_rows
    )
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "run": {
            "run_id": manifest["run_id"],
            "run_dir": str(original_run_dir or path),
            "model": manifest.get("controller", {}).get("model"),
            "reasoning_effort": manifest.get("controller", {}).get("reasoning_effort"),
            "thinking_type": manifest.get("controller", {}).get("thinking_type"),
            "provider_extra_body": manifest.get("controller", {}).get(
                "provider_extra_body", {}
            ),
            "seed": config.seed,
            "max_days": config.max_days,
            "environment_rules": manifest.get("environment_rules", environment.resolved_rules()),
        },
        "headline": {
            **({"warehouse_capacity": environment.warehouse_diagnostics(state)}
               if environment._v31_warehouse_capacity() else {}),
            **(
                {
                    (
                        "durable_capital"
                        if environment._v40_revolving_durable_capital()
                        else "durable_acquisition_budget"
                    ): (
                        environment._durable_acquisition_budget_summary(state)
                    )
                }
                if environment._durable_capital_control_enabled()
                else {}
            ),
            **(
                {
                    "durable_capital_exit_summary": {
                        "asset_count": len(durable_capital_exits),
                        "basis_total": round(
                            sum(
                                float(row["durable_capital_basis"] or 0.0)
                                for row in durable_capital_exits
                            ),
                            2,
                        ),
                        "cause_counts": dict(
                            sorted(
                                Counter(
                                    str(row["capital_exit_cause"])
                                    for row in durable_capital_exits
                                ).items()
                            )
                        ),
                    }
                }
                if environment._v40_revolving_durable_capital()
                else {}
            ),
            "liquidation_policy": final_value.get(
                "liquidation_policy",
                "rolling_depth" if config.liquidation_depth_enabled else "flat_nominal",
            ),
            "status": "bankrupt" if state.bankrupt else "completed" if state.done else "in_progress",
            "completed_days": len(state.daily_history),
            "decision_horizon_day": state.decision_horizon_day,
            "runoff_end_day": state.runoff_end_day,
            "completed_decision_days": (
                min(len(state.daily_history), int(state.decision_horizon_day))
                if state.decision_horizon_day is not None
                else len(state.daily_history)
            ),
            "completed_runoff_days": (
                max(0, len(state.daily_history) - int(state.decision_horizon_day))
                if state.decision_horizon_day is not None
                else 0
            ),
            "evaluation_score": evaluation_score,
            "initial_business_value": initial_value,
            "initial_official_liquidation_value": initial_official_liquidation_value,
            "initial_operating_business_value": initial_operating_value,
            "initial_contract_adjusted_economic_value": initial_contract_adjusted_economic_value,
            "initial_book_net_assets": initial_book_net_assets,
            "final_business_value": final_business_value,
            "business_value_if_season_ended_now": (
                business_value_at_stop if not state.done else None
            ),
            "business_value_at_stop": business_value_at_stop,
            "official_liquidation_value": (
                official_liquidation_value_at_stop if state.done else None
            ),
            "official_liquidation_value_at_stop": official_liquidation_value_at_stop,
            "operating_business_value_at_stop": operating_value_at_stop,
            "contract_adjusted_economic_value_at_stop": contract_adjusted_economic_value_at_stop,
            "efficiency_adjusted_value": (
                efficiency_adjusted_value_at_stop if state.done else None
            ),
            "efficiency_adjusted_value_at_stop": efficiency_adjusted_value_at_stop,
            "book_net_assets_at_stop": final_value.get(
                "book_net_assets", operating_value_at_stop
            ),
            "business_value_change": round(business_value_at_stop - initial_value, 2),
            "business_value_change_pct": round(
                100 * (business_value_at_stop - initial_value) / initial_value, 2
            ),
            "official_liquidation_value_change": round(
                official_liquidation_value_at_stop - initial_official_liquidation_value,
                2,
            ),
            "official_liquidation_value_change_pct": round(
                100
                * (
                    official_liquidation_value_at_stop
                    - initial_official_liquidation_value
                )
                / initial_official_liquidation_value,
                2,
            ),
            "stop_reason": summary.get("stop_reason"),
        },
        "financials": {
            **(
                {
                    (
                        "durable_capital"
                        if environment._v40_revolving_durable_capital()
                        else "durable_acquisition_budget"
                    ): (
                        environment._durable_acquisition_budget_summary(state)
                    )
                }
                if environment._durable_capital_control_enabled()
                else {}
            ),
            "final_cash": round(state.cash, 2),
            "deposit_liability": round(state.deposit_liability, 2),
            "total_revenue": round(state.cumulative_rental_revenue + state.cumulative_retail_revenue, 2),
            "rental_revenue": state.cumulative_rental_revenue,
            "retail_revenue": state.cumulative_retail_revenue,
            "asset_sale_proceeds": state.cumulative_asset_sale_proceeds,
            "retail_liquidation_proceeds": (
                state.cumulative_retail_liquidation_proceeds
            ),
            "retail_liquidation_book_cost": (
                state.cumulative_retail_liquidation_book_cost
            ),
            "retail_liquidation_loss": state.cumulative_retail_liquidation_loss,
            "retail_units_liquidated": state.cumulative_retail_units_liquidated,
            "total_cost": round(total_cost, 2),
            "operating_revenue": round(
                state.cumulative_rental_revenue
                + state.cumulative_retail_revenue
                + float(getattr(state, "cumulative_damage_deductions", 0.0)),
                2,
            ),
            "retail_cogs": round(sum(float(row["retail_cogs"]) for row in daily), 2),
            "rental_use_depreciation": round(
                sum(float(row["rental_use_depreciation"]) for row in daily), 2
            ),
            "operating_pnl": round(operating_pnl, 2),
            "business_pnl_after_retail_liquidation": (
                business_pnl_after_retail_liquidation
            ),
            "purchase_cost": state.cumulative_purchase_cost,
            "supplier_trial_handling_fees": (
                state.cumulative_supplier_trial_handling_fees
            ),
            "supplier_trial_usage_fees": (
                state.cumulative_supplier_trial_usage_fees
            ),
            "cleaning_cost": state.cumulative_cleaning_cost,
            "repair_cost": state.cumulative_repair_cost,
            "maintenance_cost": state.cumulative_maintenance_cost,
            "reservation_compensation": state.cumulative_reservation_compensation,
            "optional_line_credits": state.cumulative_optional_line_credits,
            "emergency_credit_principal": state.emergency_credit_principal,
            "emergency_credit_liability": state.emergency_credit_liability,
            "emergency_credit_fees": state.cumulative_emergency_credit_fees,
            "retail_holding_cost": state.cumulative_retail_holding_cost,
            "rental_holding_cost": state.cumulative_rental_holding_cost,
            "rental_idle_holding_cost": float(
                getattr(state, "cumulative_rental_idle_holding_cost", 0.0)
            ),
            "liquidation_nominal_proceeds": round(
                sum(
                    _number(row.get("financial", {}), "liquidation_nominal_proceeds")
                    for row in state.daily_history
                ),
                2,
            ),
            "liquidation_depth_discount": round(
                sum(
                    _number(row.get("financial", {}), "liquidation_depth_discount")
                    for row in state.daily_history
                ),
                2,
            ),
            "operating_fees": state.cumulative_fees,
            "model_cost": state.cumulative_model_cost,
            "standardized_model_cost": standardized_model_cost,
            "standardized_input_token_cost_per_million": standardized_input_rate,
            "standardized_output_token_cost_per_million": standardized_output_rate,
            "model_cost_charged_to_cash": model_cost_charged_to_cash,
            "postseason_cleaning_cost": final_value.get(
                "postseason_cleaning_cost", final_value.get("pending_cleaning_cost", 0.0)
            ),
            "postseason_cleaning_work_points": final_value.get(
                "postseason_cleaning_work_points", 0.0
            ),
        },
        "demand": {
            "requested_units": state.cumulative_requested_units,
            "fulfilled_units": state.cumulative_fulfilled_units,
            "outside_rejections": state.cumulative_outside_rejections,
            "stockout_rejections": state.cumulative_stockout_rejections,
            "service_closed_rejections": service_closed_rejections,
            "capacity_policy_rejections": int(
                getattr(state, "cumulative_capacity_policy_rejections", 0)
            ),
            "horizon_rejections": state.cumulative_horizon_rejections,
            "unpriced_rejections": int(
                getattr(state, "cumulative_unpriced_rejections", 0)
            ),
            "eligible_requested_units": eligible_requested,
            "eligible_fulfillment_rate": (
                round(state.cumulative_fulfilled_units / eligible_requested, 6)
                if eligible_requested
                else None
            ),
            "operational_requested_units": operational_requested,
            "operational_fulfillment_rate": (
                round(state.cumulative_fulfilled_units / operational_requested, 6)
                if operational_requested
                else None
            ),
            "full_baskets": state.cumulative_full_baskets,
            "partial_baskets": state.cumulative_partial_baskets,
            "unfilled_baskets": state.cumulative_unfilled_baskets,
            "operationally_evaluable_baskets": (
                state.cumulative_operationally_evaluable_baskets
            ),
            "operational_full_baskets": state.cumulative_operational_full_baskets,
            "operational_partial_baskets": state.cumulative_operational_partial_baskets,
            "operational_unfilled_baskets": state.cumulative_operational_unfilled_baskets,
            "analyst_only_internal_channel_counts": dict(
                sorted(internal_channel_counts.items())
            ),
            "analyst_only_basket_intent_status_counts": dict(
                sorted(basket_intent_status_counts.items())
            ),
        },
        "operations": {
            "customers": state.cumulative_customers,
            "transactions": state.cumulative_transactions,
            "orders": len(state.orders),
            "rental_contracts": len(state.contracts),
            "final_retail_units": sum(row["units"] for row in retail),
            "operating_rental_assets": sum(
                row.status not in {"retired", "sold"} for row in state.assets
            ),
            "assets_sold": state.cumulative_assets_sold,
            "retail_units_liquidated": state.cumulative_retail_units_liquidated,
            "rental_asset_records": len(state.assets),
            "maintenance_actions": state.cumulative_maintenance_actions,
            "decision_horizon_day": state.decision_horizon_day,
            "runoff_end_day": state.runoff_end_day,
        },
        "agent_usage": {
            "model_calls": len(models),
            "input_tokens": sum(row["input_tokens"] for row in models),
            "output_tokens": sum(row["output_tokens"] for row in models),
            "tool_calls": len(tools),
            "tool_errors": sum(not row["ok"] for row in tools),
            "agent_tool_calls": sum(
                row.get("origin") == "agent" for row in tools
            ),
            "automatic_tool_calls": sum(
                row.get("origin") == "automatic" for row in tools
            ),
            "unknown_origin_tool_calls": sum(row.get('origin') == 'unknown' for row in tools),
            "unknown_model_associations": sum(row.get('identity_status') == 'unknown' for row in models),
            "model_identity_status_counts": dict(Counter(row.get('identity_status') for row in models)),
            "agent_tool_errors": sum(
                row.get("origin") == "agent" and not row["ok"]
                for row in tools
            ),
            "automatic_tool_errors": sum(
                row.get("origin") == "automatic" and not row["ok"]
                for row in tools
            ),
            "invalid_action_episodes": sum(
                row.get("invalid_action_episode", 0) for row in models
            ),
            "business_tool_error_records": len(business_error_tools),
            "business_tool_errors_by_effect": dict(
                sorted(business_errors_by_effect.items())
            ),
            "structured_quote_validation_batch_records": len(
                structured_quote_validation_rows
            ),
            "structured_quote_validation_invalid_items": sum(
                int(row.get("data", {}).get("invalid_item_count", 0))
                for row in structured_quote_validation_rows
            ),
            "rental_capacity_tool_calls": len(capacity_tool_rows),
            "rental_capacity_tool_errors": sum(
                not row.get("ok") for row in capacity_tool_rows
            ),
            "rental_capacity_in_transit_actual_delivery_violations": (
                capacity_delivery_privacy_violations
            ),
            "walk_in_capacity_limit_tool_calls": len(
                capacity_limit_tool_rows
            ),
            "walk_in_capacity_limit_tool_errors": sum(
                not row.get("ok") for row in capacity_limit_tool_rows
            ),
            "walk_in_capacity_limit_warning_records": sum(
                len(row.get("data", {}).get("warnings", []))
                for row in capacity_limit_tool_rows
                if row.get("ok")
            ),
            "reservation_view_result_characters_by_view": {
                view: {
                    "count": len(values),
                    "median": sorted(values)[len(values) // 2],
                    "maximum": max(values),
                }
                for view, values in sorted(
                    reservation_view_result_characters.items()
                )
            },
            "generation_truncations": sum(
                row.get("generation_truncated", False) for row in models
            ),
            "generation_truncated_tool_json": sum(
                bool(row.get("generation_truncated_tool_call_ids"))
                for row in models
            ),
            "generation_truncated_tool_error_records": sum(
                row.get("generation_truncation") and not row["ok"]
                for row in tools
            ),
            "finish_reason_counts": dict(sorted(finish_reason_counts.items())),
            "context_message_characters": {
                "minimum": (
                    context_character_values[0]
                    if context_character_values
                    else 0
                ),
                "median": (
                    context_character_values[len(context_character_values) // 2]
                    if context_character_values
                    else 0
                ),
                "maximum": (
                    context_character_values[-1]
                    if context_character_values
                    else 0
                ),
            },
            "sliding_window_requests": sum(
                value > 0 for value in omitted_message_values
            ),
            "maximum_sliding_window_omitted_messages": max(
                omitted_message_values, default=0
            ),
            "generation_truncation_notice_requests": sum(
                bool(row.get("generation_truncation_notice_present"))
                for row in models
            ),
            "tool_result_characters": sum(
                int(row.get("result_characters", 0)) for row in tools
            ),
            "maximum_tool_result": (
                {
                    "tool": largest_tool_result["name"],
                    "call_id": largest_tool_result["call_id"],
                    "day": largest_tool_result["day"],
                    "characters": largest_tool_result["result_characters"],
                }
                if largest_tool_result is not None
                else None
            ),
            "duplicate_read_only_result_records": (
                duplicate_read_only_result_records
            ),
            "duplicate_read_only_result_characters": (
                duplicate_read_only_result_characters
            ),
            "repeated_read_only_while_still_in_context_records": int(
                read_only_result_reuse_audit[
                    "repeated_while_still_in_context"
                ]["records"]
            ),
            "refetched_read_only_after_context_eviction_records": int(
                read_only_result_reuse_audit[
                    "refetched_after_context_eviction"
                ]["records"]
            ),
            "forced_day_ends": state.forced_day_ends,
            "parallel_tool_call_anomalies": state.parallel_tool_call_anomalies,
            "discarded_tool_calls": sum(row["discarded_tool_calls"] for row in models),
            "read_only_tool_batches": sum(
                row["read_only_tool_batches"] for row in models
            ),
            "batched_read_only_tool_calls": sum(
                row["batched_read_only_tool_calls"] for row in models
            ),
            "rejected_tool_batches": sum(
                row["rejected_tool_batches"] for row in models
            ),
            "rejected_batched_tool_calls": sum(
                row["rejected_batched_tool_calls"] for row in models
            ),
        },
        "memory_usage": memory_usage,
        "reservations": reservation_summary,
        "reservation_protocol_audit": reservation_protocol_audit,
        "quote_history_query_audit": quote_history_query_audit,
        "quote_workspace_audit": quote_workspace_audit,
        "read_only_result_reuse_audit": read_only_result_reuse_audit,
        "database_integrity": integrity,
        "final_business_value_components": final_value,
        "daily": daily,
        "service_daily": service_daily,
        "service_summary": services,
        "product_fulfillment_daily": product_fulfillment_daily,
        "product_fulfillment_summary": product_fulfillment,
        "offer_fulfillment_daily": offer_fulfillment_daily,
        "offer_fulfillment_summary": offer_fulfillment,
        "product_weather_contribution": product_weather_contribution,
        # Alias the descriptive table with the terminology used in the
        # environment-design note; both keys point to the same observations.
        "gross_margin_by_product_and_weather": product_weather_contribution,
        "offer_concentration": offer_concentration,
        "service_concentration": service_concentration,
        "rental_activity": rental_activity,
        "rental_activity_summary": rental_activity_summary,
        "purchased_rental_contribution_by_offer": (
            purchased_rental_contribution_by_offer
        ),
        "purchased_rental_contribution_by_product": (
            purchased_rental_contribution_by_product
        ),
        "review_flags": {
            "purchased_rental_offers_above_40pct_positive_contribution": (
                concentrated_purchased_rental_offers
            ),
        },
        "counter_interactions": _counter_interaction_rows(state),
        "demand_cube": _demand_cube_rows(state),
        "internal_channel_outcomes": internal_channel_rows,
        "internal_basket_intents": internal_basket_intent_rows,
        "final_retail_inventory": retail,
        "final_rental_assets": assets,
        "supplier_trials": supplier_trials,
        "durable_capital_exits": durable_capital_exits,
        "asset_sales": asset_sales,
        "retail_liquidations": retail_liquidations,
        "reservation_requests": reservation_requests,
        "reservation_lines": reservation_lines,
        "reservation_events": reservation_events,
        "maintenance_events": maintenance_events,
        "tool_usage": tool_usage,
        "memory_documents": memory_documents,
        "files": [],
    }
    report['analysis_snapshot'] = snapshot_metadata
    unknown_models = report['agent_usage']['unknown_model_associations']
    report['agent_usage']['identity_metrics_complete'] = not unknown_models
    if unknown_models:
        # Attribution-dependent totals cannot be reported as complete zeroes.
        for key in ('invalid_action_episodes','read_only_tool_batches','batched_read_only_tool_calls',
                    'rejected_tool_batches','rejected_batched_tool_calls'):
            report['agent_usage'][key] = None
        report['agent_usage']['identity_warning'] = 'Some model/execution links are unknown; derived totals are unavailable.'
        for row in daily:
            if not row.get('identity_metrics_complete', True):
                for key in ('invalid_action_episodes','read_only_tool_batches','batched_read_only_tool_calls',
                            'rejected_tool_batches','rejected_batched_tool_calls'):
                    row[key] = None
    output = Path(output_dir).expanduser().resolve() if output_dir else path / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "daily_metrics.csv": daily,
        "service_daily_metrics.csv": service_daily,
        "service_summary.csv": services,
        "product_fulfillment_daily_metrics.csv": product_fulfillment_daily,
        "product_fulfillment_summary.csv": product_fulfillment,
        "offer_fulfillment_daily_metrics.csv": offer_fulfillment_daily,
        "offer_fulfillment_summary.csv": offer_fulfillment,
        "product_weather_contribution.csv": product_weather_contribution,
        "offer_concentration.csv": offer_concentration,
        "service_concentration.csv": service_concentration,
        "rental_activity.csv": rental_activity,
        "rental_activity_summary.csv": rental_activity_summary,
        "purchased_rental_contribution_by_offer.csv": (
            purchased_rental_contribution_by_offer
        ),
        "purchased_rental_contribution_by_product.csv": (
            purchased_rental_contribution_by_product
        ),
        "counter_interactions.csv": report["counter_interactions"],
        "demand_cube.csv": report["demand_cube"],
        "internal_channel_outcomes.csv": internal_channel_rows,
        "internal_basket_intents.csv": internal_basket_intent_rows,
        "orders.csv": _order_rows(environment, state),
        "rental_contracts.csv": _contract_rows(environment, state),
        "asset_sales.csv": asset_sales,
        "retail_liquidations.csv": retail_liquidations,
        "reservation_requests.csv": reservation_requests,
        "reservation_lines.csv": reservation_lines,
        "reservation_events.csv": reservation_events,
        "maintenance_events.csv": maintenance_events,
        "final_retail_inventory.csv": retail,
        "final_rental_assets.csv": assets,
        "supplier_trials.csv": supplier_trials,
        "durable_capital_exits.csv": durable_capital_exits,
        "tool_usage.csv": tool_usage,
        "memory_documents.csv": memory_documents,
    }
    for name, rows in tables.items():
        _write_csv(output / name, rows)
    # Lightweight evaluator only. Historical replay and local branches are never
    # launched from analyze; use the explicit evaluate --extended command.
    if daily:
        from .evaluation import FORMAT as evaluation_format
        from .evaluation.api import read_inputs
        from .evaluation.basic import basic_metrics
        from .evaluation.common import PARAMETERS, digest, implementation_digest
        from .evaluation.report import save_report
        raw_state, raw_calls, profiles = read_inputs(path)
        cutoff = max(int(row['day']) for row in daily)
        evaluation = {
            'format': evaluation_format, 'run_id': manifest['run_id'],
            'through_day': cutoff, 'protocol': PARAMETERS,
            'input_identity': {'state': digest(raw_state), 'evaluation_code': implementation_digest()},
            'environment_identity': manifest.get('controller', {}).get('implementation_identity'),
            'basic': basic_metrics(raw_state, raw_calls, cutoff, manifest, profiles),
            'extended': {'status': 'not_enabled'}, 'warnings': [],
        }
        evaluation_dir = output / 'evaluation' / evaluation_format / f'day-{cutoff}'
        # Do not replace previously computed extended results with a basic scan.
        if (evaluation_dir / 'metrics.json').exists():
            evaluation_dir = evaluation_dir / 'basic'
        save_report(evaluation, evaluation_dir, plots=False)
        report['trajectory_evaluation'] = {
            'format': evaluation_format, 'closed_day_cutoff': cutoff,
            'extended_status': 'not_enabled', 'output_dir': str(evaluation_dir),
            'service_reliability': evaluation['basic']['service']['reliability'],
            'capital_productivity': evaluation['basic']['capital_productivity'],
        }
    report["files"] = ["REPORT.md", "report.json", *tables]
    if plots and daily:
        _render_plots(report, output, dpi)
        report["files"].extend(PLOT_FILES)
    (output / "REPORT.md").write_text(_markdown(report), encoding="utf-8")
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"run_id": manifest["run_id"], "output_dir": str(output), "headline": report["headline"], "files": report["files"]}
