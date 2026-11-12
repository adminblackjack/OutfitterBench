from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .llm import ModelProvider

from .config import OutfitterConfig
from .assortment import benchmark_track_id
from .data import OutfitterDataset
from .environment import OutfitterEnvironment
from .models import ToolResult, WorldState
from .prompts import (
    continuation_message,
    day_message,
    generation_truncation_message,
    system_prompt,
)
from .store import RunStore
from .tools import ToolExecutor


RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
FIRST_ONLY_PROTOCOL = "first_only"
READ_ONLY_BATCH_PROTOCOL = "read_only_batch_v1"
ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v12-compact-ids-pagination"
TOOL_INTERFACE_VERSION = 10


def _environment_protocol(dataset: OutfitterDataset) -> str:
    lifecycle = dict(dataset.metadata.get("lifecycle_economics") or {})
    if lifecycle.get("enabled_config_required"):
        return str(dataset.metadata.get("environment_protocol") or ENVIRONMENT_PROTOCOL)
    return ENVIRONMENT_PROTOCOL


def _tool_interface_version(dataset: OutfitterDataset) -> int:
    return int(dataset.metadata.get("tool_interface_version", TOOL_INTERFACE_VERSION))


def _tool_schema_sha256(definitions: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        definitions,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _core_release_id(dataset: OutfitterDataset) -> str:
    return str(
        dataset.metadata.get("core_release_id")
        or dataset.metadata.get("environment_protocol")
        or ENVIRONMENT_PROTOCOL
    )


def _runtime_data_sha256(dataset: OutfitterDataset) -> str:
    rows = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(dataset.root.iterdir())
        if path.is_file()
    }
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _implementation_identity() -> dict[str, Any]:
    package = Path(__file__).resolve().parent
    top_level = {
        "acquisition_budget.py", "assortment.py", "channel_demand.py", "config.py", "data.py",
        "demand.py", "environment.py", "errors.py", "jsonio.py",
        "lifecycle_economics.py", "llm.py", "models.py", "prompts.py",
        "provider_schemas.py",
        "reference_demand.py", "reservation_choice.py",
        "reservation_contingency.py",
        "revolving_durable_capital.py",
        "contingency_actor_interface.py",
        "reservation_feedback.py", "reservation_profiles.py", "runner.py",
        "service_commitments.py", "store.py", "supplier_trial.py", "tool_executor.py",
        "tool_specs.py", "v5.py", "warehouse_capacity.py",
        "wide_date_flexibility.py", "capital_balance_tuning.py",
        "weather_information.py",
        "trip_weather.py",
        "reservation_difficulty.py",
        "return_dynamics.py",
        "autonomous_rentals.py",
        "service_guarantee.py",
        "school_holiday.py",
        "summer_pressure.py", "reservation_overlap.py", "resale_disclosure.py",
        "operating_history.py",
        "actor_contract_v53.py",
        "recovery_state.py",
        "call_identity.py",
        "actor_manual.py", "actor_records.py", "actor_tool_text.py",
    }
    paths = [package / name for name in sorted(top_level)]
    paths.extend(sorted((package / "runtime").rglob("*.py")))
    rows = {
        path.relative_to(package).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
    }
    return {
        "schema_version": "outfitter-core-source-identity-v1",
        "source_file_count": len(rows),
        "source_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
    }


def generate_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"outfitter-{stamp}-{uuid.uuid4().hex[:8]}"


class OutfitterRunner:
    def __init__(
        self,
        *,
        config: OutfitterConfig,
        provider: ModelProvider,
        store: RunStore,
        dataset: OutfitterDataset,
        tool_call_protocol: str,
    ):
        self.config = config
        self.provider = provider
        self.store = store
        self.dataset = dataset
        self.tool_call_protocol = tool_call_protocol
        self.environment = OutfitterEnvironment(config, dataset)
        self.tools = ToolExecutor(self.environment)
        self.system_prompt = system_prompt(
            self.environment.public_operating_rules(),
            tool_call_protocol=tool_call_protocol,
            read_only_tools=self.tools.read_only_names(),
            max_read_only_batch_size=config.max_read_only_batch_size,
        )

    @classmethod
    def create(
        cls,
        *,
        config: OutfitterConfig,
        provider: ModelProvider,
        run_id: str | None = None,
    ) -> "OutfitterRunner":
        config.validate()
        resolved_id = run_id or generate_run_id()
        if not RUN_ID_RE.fullmatch(resolved_id):
            raise ValueError("run_id must use letters, digits, dot, underscore, or dash")
        dataset = OutfitterDataset.load(config.data_dir)
        dataset.require_weather_coverage(
            config.max_days + config.runoff_days,
            start_day=config.weather_start_day,
        )
        environment = OutfitterEnvironment(config, dataset)
        tools = ToolExecutor(environment)
        environment_protocol = _environment_protocol(dataset)
        tool_call_protocol = (
            READ_ONLY_BATCH_PROTOCOL
            if config.parallel_tool_calls
            else FIRST_ONLY_PROTOCOL
        )
        tool_definitions = tools.definitions()
        manual = tools.business_rules_snapshot
        manual_text = (
            json.dumps(manual, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            if manual is not None else None
        )
        manifest = {
            "format": "outfitter-run-v1",
            "run_id": resolved_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": config.to_dict(),
            "runtime_data": "data",
            "system_prompt": "system_prompt.md",
            **({"business_rules": "business_rules.json",
                "business_rules_sha256": hashlib.sha256(manual_text.encode()).hexdigest()}
               if manual_text is not None else {}),
            "controller": {
                "environment_protocol": environment_protocol,
                **({"business_manual_policy": dataset.metadata["business_manual_policy"]}
                   if manual is not None else {}),
                **({"workspace_budget_message_policy": "append-only-tail-v1"}
                   if environment._v43_actor_manual() else {}),
                "core_release_id": _core_release_id(dataset),
                # Provenance only: not part of the actor's observations or rules.
                **({"weather_dataset_id": dataset.metadata["weather_dataset_id"]}
                   if dataset.metadata.get("weather_dataset_id") else {}),
                "assortment_mode": config.assortment_mode,
                "benchmark_track_id": benchmark_track_id(config.assortment_mode),
                **(
                    {
                        "durable_acquisition_budget_policy": dataset.metadata[
                            "durable_acquisition_budget_policy"
                        ]
                    }
                    if dataset.metadata.get("durable_acquisition_budget_policy")
                    else {}
                ),
                **(
                    {
                        "supplier_trial_policy": dataset.metadata[
                            "supplier_trial_policy"
                        ]
                    }
                    if dataset.metadata.get("supplier_trial_policy")
                    else {}
                ),
                **(
                    {
                        "proactive_contingency_policy": dataset.metadata[
                            "proactive_contingency_policy"
                        ]
                    }
                    if dataset.metadata.get("proactive_contingency_policy")
                    else {}
                ),
                **(
                    {
                        "contingency_actor_interface_policy": dataset.metadata[
                            "contingency_actor_interface_policy"
                        ]
                    }
                    if dataset.metadata.get("contingency_actor_interface_policy")
                    else {}
                ),
                **(
                    {
                        "actor_interface_slimming_policy": dataset.metadata[
                            "actor_interface_slimming_policy"
                        ]
                    }
                    if dataset.metadata.get("actor_interface_slimming_policy")
                    else {}
                ),
                "runtime_data_sha256": _runtime_data_sha256(dataset),
                "implementation_identity": _implementation_identity(),
                "tool_interface_version": _tool_interface_version(dataset),
                "tool_schema_sha256": _tool_schema_sha256(tool_definitions),
                "model": config.model,
                "reasoning_effort": config.reasoning_effort,
                "thinking_type": config.thinking_type,
                "provider_extra_body": config.provider_extra_body,
                "parallel_tool_calls": config.parallel_tool_calls,
                "tool_call_protocol": tool_call_protocol,
                "max_read_only_batch_size": config.max_read_only_batch_size,
                "tool_effects": tools.effect_map(),
            },
            "environment_rules": environment.resolved_rules(),
            "tool_names": [row["function"]["name"] for row in tool_definitions],
        }
        store = RunStore.create(
            Path(config.runs_dir).expanduser().resolve() / resolved_id,
            manifest,
            environment.reset(),
        )
        dataset.snapshot_to(store.run_dir / "data")
        runner = cls(
            config=config,
            provider=provider,
            store=store,
            dataset=dataset,
            tool_call_protocol=tool_call_protocol,
        )
        (store.run_dir / "system_prompt.md").write_text(
            runner.system_prompt, encoding="utf-8"
        )
        if manual_text is not None:
            (store.run_dir / "business_rules.json").write_text(manual_text, encoding="utf-8")
        return runner

    @classmethod
    def resume(cls, *, run_dir: str | Path, provider: ModelProvider) -> "OutfitterRunner":
        store = RunStore.resume(run_dir)
        manifest = json.loads((store.run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != "outfitter-run-v1":
            raise ValueError("unsupported run format")
        values = dict(manifest["config"])
        values["data_dir"] = str(store.run_dir / manifest.get("runtime_data", "data"))
        config = OutfitterConfig.from_dict(values)
        dataset = OutfitterDataset.load(config.data_dir)
        dataset.require_weather_coverage(
            config.max_days + config.runoff_days,
            start_day=config.weather_start_day,
        )
        tools = ToolExecutor(OutfitterEnvironment(config, dataset))
        tool_definitions = tools.definitions()
        expected_tools = [row["function"]["name"] for row in tool_definitions]
        if expected_tools != manifest.get("tool_names"):
            raise ValueError("run tool interface is incompatible with this code")
        controller = manifest.get("controller", {})
        if controller.get("environment_protocol") != _environment_protocol(dataset):
            raise ValueError("run environment protocol is incompatible with this code")
        recorded_runtime = controller.get("runtime_data_sha256")
        if recorded_runtime is not None and recorded_runtime != _runtime_data_sha256(dataset):
            raise ValueError("run runtime data is incompatible with this code")
        recorded_source = controller.get("implementation_identity")
        if recorded_source is not None and recorded_source != _implementation_identity():
            raise ValueError("run implementation source is incompatible with this code")
        if controller.get("core_release_id", _core_release_id(dataset)) != _core_release_id(dataset):
            raise ValueError("run core release is incompatible with this code")
        if controller.get("assortment_mode", "rental_and_retail") != config.assortment_mode:
            raise ValueError("run assortment mode is incompatible with this code")
        recorded_budget_policy = controller.get(
            "durable_acquisition_budget_policy"
        )
        current_budget_policy = dataset.metadata.get(
            "durable_acquisition_budget_policy"
        )
        if recorded_budget_policy != current_budget_policy:
            raise ValueError(
                "run durable acquisition budget policy is incompatible with this code"
            )
        if controller.get("supplier_trial_policy") != dataset.metadata.get(
            "supplier_trial_policy"
        ):
            raise ValueError(
                "run supplier trial policy is incompatible with this code"
            )
        if controller.get("proactive_contingency_policy") != dataset.metadata.get(
            "proactive_contingency_policy"
        ):
            raise ValueError(
                "run proactive contingency policy is incompatible with this code"
            )
        if controller.get(
            "contingency_actor_interface_policy"
        ) != dataset.metadata.get("contingency_actor_interface_policy"):
            raise ValueError(
                "run contingency actor-interface policy is incompatible with this code"
            )
        if controller.get(
            "benchmark_track_id", benchmark_track_id(config.assortment_mode)
        ) != benchmark_track_id(config.assortment_mode):
            raise ValueError("run benchmark track is incompatible with this code")
        if controller.get("tool_interface_version") != _tool_interface_version(dataset):
            raise ValueError("run tool schema is incompatible with this code")
        recorded_schema_hash = controller.get("tool_schema_sha256")
        if (
            recorded_schema_hash is not None
            and recorded_schema_hash != _tool_schema_sha256(tool_definitions)
        ):
            raise ValueError("run tool schema hash is incompatible with this code")
        tool_call_protocol = controller.get("tool_call_protocol", FIRST_ONLY_PROTOCOL)
        if tool_call_protocol not in {FIRST_ONLY_PROTOCOL, READ_ONLY_BATCH_PROTOCOL}:
            raise ValueError("run tool-call protocol is incompatible with this code")
        if tool_call_protocol == READ_ONLY_BATCH_PROTOCOL:
            if controller.get("tool_effects") != tools.effect_map():
                raise ValueError("run tool effects are incompatible with this code")
            if int(controller.get("max_read_only_batch_size", 0)) != config.max_read_only_batch_size:
                raise ValueError("run read-only batch limit is incompatible with this code")
        runner = cls(
            config=config,
            provider=provider,
            store=store,
            dataset=dataset,
            tool_call_protocol=tool_call_protocol,
        )
        prompt_path = store.run_dir / manifest.get("system_prompt", "system_prompt.md")
        if prompt_path.is_file():
            runner.system_prompt = prompt_path.read_text(encoding="utf-8")
        if runner.environment._v43_actor_manual():
            if controller.get("business_manual_policy") != dataset.metadata.get("business_manual_policy"):
                raise ValueError("run public business manual policy is incompatible with this code")
            if manifest.get("business_rules") != "business_rules.json":
                raise ValueError("run public business rules snapshot is missing")
            manual_path = store.run_dir / "business_rules.json"
            if not manual_path.is_file():
                raise ValueError("run public business rules snapshot is missing")
            manual_text = manual_path.read_text(encoding="utf-8")
            if hashlib.sha256(manual_text.encode()).hexdigest() != manifest.get("business_rules_sha256"):
                raise ValueError("run public business rules snapshot failed integrity check")
            manual = json.loads(manual_text)
            if manual != runner.tools.business_rules_snapshot:
                raise ValueError("run public business rules differ from the configured contract")
            runner.tools.business_rules_snapshot = manual
        return runner

    def run(self, *, stop_after_model_calls: int | None = None,
            stop_after_day: int | None = None) -> dict[str, Any]:
        if stop_after_day is not None and (type(stop_after_day) is not int or stop_after_day < 1):
            raise ValueError("stop_after_day must be a positive integer")
        starting_calls = self.store.model_call_count()
        call_limit = self.config.max_model_calls
        if stop_after_model_calls is not None:
            call_limit = min(call_limit, starting_calls + stop_after_model_calls)
        stop_reason = "completed"
        while True:
            state = self.store.load_state()
            if state.done:
                break
            if stop_after_day is not None and state.day > stop_after_day:
                stop_reason = "decision_day_limit"
                break
            if state.day > self.config.max_days:
                call_id = f"automatic-runoff-close-day-{state.day}"
                state, result, _ = self.store.execute_tool(
                    call_id,
                    "close_business_day",
                    {},
                    self.tools.execute,
                    controller_kind="runoff",
                )
                self.store.record_tool_message(
                    state,
                    tool_call_id=call_id,
                    name="close_business_day",
                    content=self.tools.encode_result(result),
                )
                continue
            if self.store.model_call_count() >= call_limit:
                stop_reason = "model_call_limit"
                break
            if not self.store.has_day_start(state.day):
                observation = self.environment.observation(state, consume_inbox=True)
                self.store.record_day_start(state, {"role": "user", "content": day_message(observation)})
            if not self._run_day(call_limit):
                stop_reason = "model_call_limit"
                break
        state = self.store.load_state()
        summary = self.summary(state, stop_reason)
        self.store.write_summary(summary)
        return summary

    def _run_day(self, call_limit: int) -> bool:
        day = self.store.load_state().day
        while self.store.model_calls_for_day(day) < self.config.max_iterations_per_day:
            terminal = self._recover_or_process_pending_calls()
            state = self.store.load_state()
            if terminal or state.done:
                return True
            if self.store.model_call_count() >= call_limit:
                return False
            messages = self._build_context(state)
            request: dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "tools": self.tools.definitions(),
                "parallel_tool_calls": self.config.parallel_tool_calls,
            }
            if self.config.reasoning_effort is not None:
                request["reasoning_effort"] = self.config.reasoning_effort
            if self.config.provider_extra_body:
                request["extra_body"] = self.config.provider_extra_body
            response = self.provider.complete(messages=messages, tools=self.tools.definitions())
            assistant_message, discarded = self._normalize_assistant_message(
                response.assistant_message
            )
            assistant_message = self._uniquify_tool_call_ids(assistant_message)
            finish_reason = self._response_finish_reason(response.raw)
            malformed_truncated_tools: list[str] = []
            if finish_reason == "length":
                assistant_message.pop("reasoning_content", None)
                malformed_truncated_tools = self._malformed_tool_names(
                    list(assistant_message.get("tool_calls") or [])
                )
                if malformed_truncated_tools:
                    assistant_message.pop("tool_calls", None)
                assistant_message["_finish_reason"] = "length"
            if discarded:
                # Some OpenAI-compatible endpoints ignore parallel_tool_calls=false.
                state.parallel_tool_call_anomalies += 1
                state.discarded_tool_calls += discarded
                assistant_message["_discarded_tool_calls"] = discarded
            calls = list(assistant_message.get("tool_calls") or [])
            if self.tool_call_protocol == READ_ONLY_BATCH_PROTOCOL and len(calls) > 1:
                batch_error = self._tool_batch_error(calls)
                if batch_error is None:
                    state.read_only_tool_batches += 1
                    state.batched_read_only_tool_calls += len(calls)
                else:
                    state.rejected_tool_batches += 1
                    state.rejected_batched_tool_calls += len(calls)
            self.store.record_model_response(
                state,
                request=request,
                response=response.raw,
                assistant_message=assistant_message,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                request_tail_message=(
                    messages[-1] if self.environment._v43_actor_manual() else None
                ),
                charge=lambda current, inputs, outputs: self.environment.charge_model_usage(
                    current, inputs, outputs
                ),
            )
            if not assistant_message.get("tool_calls"):
                notice = (
                    generation_truncation_message(
                        malformed_tool_names=malformed_truncated_tools
                    )
                    if finish_reason == "length"
                    else continuation_message("the previous response used no tool")
                )
                self.store.append_message(
                    day,
                    "controller",
                    {"role": "user", "content": notice},
                )
        # A response at the per-day limit has already been charged and persisted.
        # Its retained tool call must receive the same execution opportunity as
        # every earlier response before the controller considers a forced close.
        terminal = self._recover_or_process_pending_calls()
        state = self.store.load_state()
        if terminal or state.done or state.day != day:
            return True
        call_id = f"forced-close-day-{day}"
        state, result, _ = self.store.execute_tool(call_id, "close_business_day", {}, self.tools.execute,
                                                  controller_kind="forced_close")
        self.store.record_tool_message(
            state, tool_call_id=call_id, name="close_business_day", content=self.tools.encode_result(result)
        )
        state.forced_day_ends += 1
        # Persist the counter with a tiny internal no-op write through the store state API.
        with self.store._connect() as connection:
            self.store._save(connection, state)
        return True

    def _recover_or_process_pending_calls(self) -> bool:
        messages = self.store.messages()
        assistant_index = None
        for index in range(len(messages) - 1, -1, -1):
            role = messages[index].get("role")
            if role == "assistant":
                if messages[index].get("tool_calls"):
                    assistant_index = index
                break
        if assistant_index is None:
            return False
        assistant = messages[assistant_index]
        following = messages[assistant_index + 1 :]
        if any(row.get("role") not in {"tool"} for row in following):
            return False
        returned = {str(row.get("tool_call_id")) for row in following}
        calls = assistant.get("tool_calls") or []
        if all(str(call.get("id")) in returned for call in calls):
            if self._tool_batch_error(calls) is not None:
                return False
            return any(call.get("function", {}).get("name") == "close_business_day" for call in calls)

        batch_error = self._tool_batch_error(calls)
        terminal = False
        for call in calls:
            call_id = str(call.get("id") or f"missing-{uuid.uuid4().hex}")
            if call_id in returned:
                continue
            if self.store.tool_call_id_taken(call_id):
                call_id = self._allocate_tool_call_id(call_id)
                if not self.store.rewrite_last_assistant_tool_call_id(
                    str(call.get("id") or ""),
                    call_id,
                ):
                    raise RuntimeError(
                        f"failed to rewrite colliding tool call id: {call.get('id')}"
                    )
                call["id"] = call_id
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            raw = function.get("arguments", "{}")
            try:
                arguments = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                arguments = {"_malformed_json": str(raw)}
            if not isinstance(arguments, dict):
                arguments = {"_non_object_arguments": arguments}
            if batch_error is not None:
                execute = lambda state, tool, args: ToolResult(
                    False,
                    {
                        "error": "tool_batch_policy_violation",
                        "detail": batch_error,
                        "executed": False,
                        "batch_size": len(calls),
                        "tool": tool,
                    },
                )
            elif terminal:
                execute = lambda state, tool, args: ToolResult(
                    False,
                    {"error": "not_processed_after_business_day_closed", "tool": tool},
                )
            else:
                execute = self.tools.execute
            state, result, _ = self.store.execute_tool(call_id, name, arguments, execute)
            self.store.record_tool_message(
                state,
                tool_call_id=call_id,
                name=name,
                content=self.tools.encode_result(result),
            )
            terminal = terminal or result.terminal_day
        discarded = int(assistant.get("_discarded_tool_calls", 0))
        if discarded:
            notice_day = state.day
            self.store.append_message(
                notice_day,
                "controller",
                {
                    "role": "user",
                    "content": (
                        "Business tool service notice: your previous response submitted "
                        f"{discarded + 1} requests together. Only the first was processed; "
                        f"the remaining {discarded} were not processed. Submit any still-needed "
                        "requests one at a time."
                    ),
                },
            )
        return terminal

    def _build_context(self, state: WorldState) -> list[dict[str, Any]]:
        persisted = [
            self._api_message(message)
            for message in self.store.context_messages_for_day(state.day)
        ]
        blocks = self._message_blocks(persisted)
        prefix = [{"role": "system", "content": self.system_prompt}]
        # Never rewrite a dynamic counter before the conversation history:
        # doing so invalidates the reusable prefix on every model request.
        suffix = []
        if self.environment._v43_actor_manual():
            suffix.append({"role": "user", "content": "Workspace budget: " + json.dumps({
                "day": state.day,
                "daily_response_limit": self.config.max_iterations_per_day,
                "remaining_responses_today_including_next": max(
                    0, self.config.max_iterations_per_day - self.store.model_calls_for_day(state.day)
                ),
                "remaining_responses_in_run_including_next": max(
                    0, self.config.max_model_calls - self.store.model_call_count()
                ),
                "daily_limit_action": "execute_last_permitted_tools_then_close_if_still_open",
                "run_limit_action": "pause_execution",
            }, sort_keys=True)})
        budget = self.config.context_budget_chars - sum(
            len(json.dumps(message, ensure_ascii=False)) for message in prefix + suffix
        )
        selected: list[list[dict[str, Any]]] = []
        used = 0
        for block in reversed(blocks):
            size = len(json.dumps(block, ensure_ascii=False))
            if selected and used + size > budget:
                break
            selected.append(block)
            used += size
        selected.reverse()
        flattened = [message for block in selected for message in block]
        omitted = len(persisted) - len(flattened)
        if omitted:
            prefix.append(self._context_omission_notice(omitted))
        return prefix + flattened + suffix

    @staticmethod
    def _context_omission_notice(omitted: int) -> dict[str, str]:
        return {
            "role": "user",
            "content": (
                f"{omitted} older messages from today no longer fit in the active "
                "workspace. Persistent memory documents are unaffected."
            ),
        }

    @staticmethod
    def _response_finish_reason(raw: dict[str, Any]) -> str | None:
        choices = raw.get("choices") if isinstance(raw, dict) else None
        if isinstance(choices, list) and choices:
            value = choices[0].get("finish_reason")
            return None if value is None else str(value)
        value = raw.get("finish_reason") if isinstance(raw, dict) else None
        return None if value is None else str(value)

    @staticmethod
    def _malformed_tool_names(calls: list[dict[str, Any]]) -> list[str]:
        malformed: list[str] = []
        for call in calls:
            function = call.get("function") or {}
            raw = function.get("arguments", "{}")
            try:
                value = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                value = None
            if not isinstance(value, dict):
                malformed.append(str(function.get("name") or "unknown_tool"))
        return malformed

    @staticmethod
    def _api_message(message: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            key: value
            for key, value in message.items()
            if key
            in {
                "role",
                "content",
                "reasoning_content",
                "tool_calls",
                "tool_call_id",
                "name",
            }
        }
        if allowed.get("role") == "assistant":
            allowed.pop("name", None)
            allowed.pop("tool_call_id", None)
        else:
            allowed.pop("reasoning_content", None)
        return allowed

    def _allocate_tool_call_id(self, original: str) -> str:
        base = original or f"missing-{uuid.uuid4().hex}"
        for _ in range(16):
            candidate = f"{base}__{uuid.uuid4().hex[:8]}"
            if not self.store.tool_call_id_taken(candidate):
                return candidate
        raise RuntimeError(f"unable to allocate a unique tool call id from {original!r}")

    def _uniquify_tool_call_ids(self, message: dict[str, Any]) -> dict[str, Any]:
        calls = list(message.get("tool_calls") or [])
        if not calls:
            return message
        uniquified = dict(message)
        rewritten: list[dict[str, Any]] = []
        used: set[str] = set()
        for call in calls:
            item = dict(call)
            candidate = str(item.get("id") or f"missing-{uuid.uuid4().hex}")
            if candidate in used or self.store.tool_call_id_taken(candidate):
                candidate = self._allocate_tool_call_id(candidate)
            item["id"] = candidate
            used.add(candidate)
            rewritten.append(item)
        uniquified["tool_calls"] = rewritten
        return uniquified

    def _normalize_assistant_message(
        self,
        message: dict[str, Any],
    ) -> tuple[dict[str, Any], int]:
        normalized = dict(message)
        calls = list(normalized.get("tool_calls") or [])
        if len(calls) <= 1 or self.tool_call_protocol == READ_ONLY_BATCH_PROTOCOL:
            if calls:
                normalized["tool_calls"] = calls
            return normalized, 0
        normalized["tool_calls"] = calls[:1]
        return normalized, len(calls) - 1

    def _tool_batch_error(self, calls: list[dict[str, Any]]) -> str | None:
        if self.tool_call_protocol != READ_ONLY_BATCH_PROTOCOL or len(calls) <= 1:
            return None
        if len(calls) > self.config.max_read_only_batch_size:
            return (
                "at most "
                f"{self.config.max_read_only_batch_size} read-only requests may be "
                "submitted together; no request in this batch was executed"
            )
        names = [str((call.get("function") or {}).get("name") or "") for call in calls]
        if not all(self.tools.is_read_only(name) for name in names):
            return (
                "multiple requests may be submitted together only when every request is "
                "read-only; state-changing requests, memory writes or deletions, and "
                "close_business_day must be submitted alone; no request in this batch "
                "was executed"
            )
        return None

    @staticmethod
    def _message_blocks(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        blocks: list[list[dict[str, Any]]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            block = [message]
            index += 1
            if message.get("role") == "assistant" and message.get("tool_calls"):
                while index < len(messages) and messages[index].get("role") == "tool":
                    block.append(messages[index])
                    index += 1
            elif message.get("role") == "tool":
                # A stored orphan is never safe to send to an API.
                continue
            blocks.append(block)
        return blocks

    def _v28_cash_flow_summary(self, state: WorldState) -> dict[str, Any]:
        recognized_prepayments = round(
            sum(
                float(row.get("financial", {}).get(
                    "reservation_prepayment_recognized", 0.0
                ))
                for row in state.daily_history
            ),
            2,
        )
        closed_days = {
            int(row["day"])
            for row in state.daily_history
            if row.get("day") is not None
        }
        if state.day not in closed_days:
            recognized_prepayments = round(
                recognized_prepayments
                + float(
                    state.current_day_ledger.get(
                        "reservation_prepayment_recognized", 0.0
                    )
                ),
                2,
            )
        rental_cash_collected = round(
            float(state.cumulative_rental_revenue)
            - recognized_prepayments
            + float(state.cumulative_reservation_prepayments),
            2,
        )
        inflows = {
            "rental_cash_collected": rental_cash_collected,
            "retail_sales_collected": round(
                float(state.cumulative_retail_revenue), 2
            ),
            "security_deposits_collected": round(
                float(state.cumulative_deposits_collected), 2
            ),
            "rental_asset_sale_proceeds": round(
                float(state.cumulative_asset_sale_proceeds), 2
            ),
            "retail_liquidation_proceeds": round(
                float(state.cumulative_retail_liquidation_proceeds), 2
            ),
            "emergency_credit_principal_drawn": round(
                float(state.emergency_credit_principal), 2
            ),
        }
        outflows = {
            "customer_refunds": round(float(state.cumulative_refunds), 2),
            "purchases": round(float(state.cumulative_purchase_cost), 2),
            **(
                {
                    "supplier_trial_handling_fees": round(
                        float(state.cumulative_supplier_trial_handling_fees), 2
                    ),
                    "supplier_trial_usage_fees": round(
                        float(state.cumulative_supplier_trial_usage_fees), 2
                    ),
                }
                if self.environment._v37_supplier_trial()
                else {}
            ),
            "operating_fees": round(float(state.cumulative_fees), 2),
            "cleaning": round(float(state.cumulative_cleaning_cost), 2),
            "repairs": round(float(state.cumulative_repair_cost), 2),
            "maintenance": round(
                float(state.cumulative_maintenance_cost), 2
            ),
            "rental_holding": round(
                float(state.cumulative_rental_holding_cost), 2
            ),
            "retail_holding": round(
                float(state.cumulative_retail_holding_cost), 2
            ),
            "reservation_compensation": round(
                float(state.cumulative_reservation_compensation), 2
            ),
            "model_usage_charged_to_cash": round(
                float(state.cumulative_model_cost)
                if self.config.charge_model_usage_to_cash
                else 0.0,
                6,
            ),
        }
        total_inflows = round(sum(inflows.values()), 6)
        total_outflows = round(sum(outflows.values()), 6)
        unpaid_shortfall = round(float(state.bankruptcy_shortfall), 2)
        expected_ending_cash = round(
            float(self.config.initial_cash)
            + total_inflows
            - total_outflows
            + unpaid_shortfall,
            6,
        )
        deposit_refunds = round(
            max(
                0.0,
                float(state.cumulative_refunds)
                - float(state.cumulative_reservation_prepayment_refunds),
            ),
            2,
        )
        damage_retained = round(
            float(state.cumulative_deposits_collected)
            - deposit_refunds
            - float(state.deposit_liability),
            2,
        )
        return {
            "basis": "gross_cash_receipts_and_payments",
            "opening_cash": round(float(self.config.initial_cash), 2),
            "inflows": inflows,
            "total_inflows": total_inflows,
            "outflows": outflows,
            "total_outflows": total_outflows,
            "unpaid_bankruptcy_shortfall_adjustment": unpaid_shortfall,
            "expected_ending_cash": expected_ending_cash,
            "ending_cash": round(float(state.cash), 6),
            "reconciliation_difference": round(
                float(state.cash) - expected_ending_cash, 6
            ),
            "deposit_settlement": {
                "security_deposits_collected": round(
                    float(state.cumulative_deposits_collected), 2
                ),
                "security_deposit_refunds": deposit_refunds,
                "ending_deposit_liability": round(
                    float(state.deposit_liability), 2
                ),
                "damage_deductions_retained": damage_retained,
                "recorded_damage_deductions": round(
                    float(state.cumulative_damage_deductions), 2
                ),
                "reconciliation_difference": round(
                    damage_retained
                    - float(state.cumulative_damage_deductions),
                    2,
                ),
                "included_separately_in_cash_inflows": False,
            },
            "noncash_credit_liability": {
                "ending_emergency_credit_liability": round(
                    float(state.emergency_credit_liability), 2
                ),
                "accrued_emergency_credit_fees": round(
                    float(state.cumulative_emergency_credit_fees), 2
                ),
            },
            "reservation_prepayment_recognized": recognized_prepayments,
        }

    def _v28_reservation_rent_cash_summary(
        self, state: WorldState
    ) -> dict[str, Any]:
        prepayment = round(
            sum(
                float(row.prepayment_collected)
                for row in state.reservation_commitments
            ),
            2,
        )
        pickup = round(
            sum(
                float(row.rent_collected_at_pickup)
                for row in state.reservation_commitments
            ),
            2,
        )
        post_pickup = round(
            sum(
                float(row.post_pickup_rent_collected)
                for row in state.reservation_commitments
            ),
            2,
        )
        refunded = round(
            sum(
                float(row.prepayment_refunded)
                for row in state.reservation_commitments
            ),
            2,
        )
        gross_cash = round(prepayment + pickup + post_pickup, 2)
        net_cash = round(gross_cash - refunded, 2)
        event_extension_cash = round(
            sum(
                float(
                    event.details.get("additional_rent_collected", 0.0)
                )
                for event in state.reservation_events
                if event.event_type in {"extension_accepted","continuation_rent_collected"}
            ),
            2,
        )
        return {
            "basis": "commitment_level_rent_cash_history",
            "commitment_count": len(state.reservation_commitments),
            "prepayment_collected": prepayment,
            "rent_collected_at_pickup": pickup,
            "post_pickup_rent_collected": post_pickup,
            "prepayment_refunded": refunded,
            "gross_rent_cash_collected": gross_cash,
            "net_rent_cash_collected": net_cash,
            "ending_outstanding_rent": round(
                sum(
                    float(row.outstanding_rent)
                    for row in state.reservation_commitments
                ),
                2,
            ),
            "ending_unearned_revenue": round(
                sum(
                    float(row.unearned_revenue)
                    for row in state.reservation_commitments
                ),
                2,
            ),
            "compensation_paid": round(
                sum(
                    float(row.compensation_paid)
                    for row in state.reservation_commitments
                ),
                2,
            ),
            "extension_rent_collected_from_events": event_extension_cash,
            "extension_event_reconciliation_difference": round(
                post_pickup - event_extension_cash, 2
            ),
            "cumulative_prepayment_reconciliation_difference": round(
                prepayment - float(state.cumulative_reservation_prepayments),
                2,
            ),
            "cumulative_refund_reconciliation_difference": round(
                refunded
                - float(state.cumulative_reservation_prepayment_refunds),
                2,
            ),
        }

    def _v28_value_change_summary(
        self, value: dict[str, Any]
    ) -> dict[str, Any]:
        initial_state = self.environment.reset()
        initial = self.environment.business_value(initial_state)
        initial_official = float(initial["official_liquidation_value"])
        current_official = float(value["official_liquidation_value"])
        initial_contract = float(initial["contract_adjusted_economic_value"])
        current_contract = float(value["contract_adjusted_economic_value"])
        return {
            "initial_official_liquidation_value": round(initial_official, 2),
            "official_liquidation_value_at_stop": round(current_official, 2),
            "official_value_change": round(
                current_official - initial_official, 2
            ),
            "official_value_change_pct": round(
                100.0 * (current_official - initial_official) / initial_official,
                6,
            )
            if initial_official
            else None,
            "initial_contract_adjusted_economic_value": round(
                initial_contract, 2
            ),
            "contract_adjusted_economic_value_at_stop": round(
                current_contract, 2
            ),
            "contract_adjusted_value_change": round(
                current_contract - initial_contract, 2
            ),
        }

    def summary(self, state: WorldState, stop_reason: str) -> dict[str, Any]:
        from .service_commitments import service_commitment_diagnostics
        value = self.environment.business_value(state)
        completed_solvent = state.done and not state.bankrupt
        v28_summary = _tool_interface_version(self.dataset) >= 24
        v29_summary = _tool_interface_version(self.dataset) >= 25
        manager_cancellations = sum(
            commitment.status == "cancelled_by_manager"
            for commitment in state.reservation_commitments
        )
        manager_cancellation_events = [
            event
            for event in state.reservation_events
            if event.event_type == "cancelled_by_manager"
        ]
        return {
            **(
                {"summary_format": "outfitter-summary-v2-auditable"}
                if v28_summary
                else {}
            ),
            **({"warehouse_capacity": self.environment.warehouse_diagnostics(state)}
               if self.environment._v31_warehouse_capacity() else {}),
            **(
                {
                    (
                        "durable_capital"
                        if self.environment._v40_revolving_durable_capital()
                        else "durable_acquisition_budget"
                    ): (
                        self.environment._durable_acquisition_budget_summary(state)
                    )
                }
                if self.environment._durable_capital_control_enabled()
                else {}
            ),
            **(
                {
                    "supplier_trial": self.environment._supplier_trial_summary(
                        state
                    )
                }
                if self.environment._v37_supplier_trial()
                else {}
            ),
            **(
                {
                    "proactive_contingency": {
                        "inquiries": state.cumulative_reservation_contingency_inquiries,
                        "applied": state.cumulative_reservation_contingencies_applied,
                        "merchant_induced_customer_cancellations": (
                            state.cumulative_reservation_contingency_customer_cancellations
                        ),
                    }
                }
                if self.environment._v38_proactive_contingency()
                else {}
            ),
            **({"service_commitments": service_commitment_diagnostics(state)}
               if self.environment._v32_service_commitments() else {}),
            "run_id": self.store.run_dir.name,
            "core_release_id": _core_release_id(self.dataset),
            "assortment_mode": self.config.assortment_mode,
            "benchmark_track_id": benchmark_track_id(
                self.config.assortment_mode
            ),
            **(
                {
                    "liquidation_policy": self.dataset.metadata.get(
                        "liquidation_policy"
                    )
                }
                if v29_summary
                else {}
            ),
            "stop_reason": stop_reason,
            **({'service_guarantee':self.environment.service_guarantee_summary(state)}
               if self.environment._v48_service_guarantee() else {}),
            "day": state.day,
            **(
                {}
                if _tool_interface_version(self.dataset) >= 23
                else {"max_days": state.max_days}
            ),
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
            "done": state.done,
            "bankrupt": state.bankrupt,
            "bankruptcy_shortfall": state.bankruptcy_shortfall,
            "evaluation_score": (
                self.environment.evaluation_score(state) if state.done else None
            ),
            "official_liquidation_value": (
                value["official_liquidation_value"] if completed_solvent else None
            ),
            "efficiency_adjusted_value": (
                value["efficiency_adjusted_value"] if completed_solvent else None
            ),
            "final_business_value": (
                value["business_value"] if completed_solvent else None
            ),
            "business_value_if_season_ended_now": (
                value["business_value"] if not state.done else None
            ),
            "going_concern_business_value": value[
                "going_concern_business_value"
            ],
            "contract_adjusted_economic_value": value[
                "contract_adjusted_economic_value"
            ],
            "book_net_assets": value["book_net_assets"],
            "business_value_breakdown": value,
            **(
                {
                    "cash_flow": self._v28_cash_flow_summary(state),
                    "reservation_rent_cash": (
                        self._v28_reservation_rent_cash_summary(state)
                    ),
                    "value_change": self._v28_value_change_summary(value),
                }
                if v28_summary
                else {}
            ),
            "cash": round(state.cash, 2),
            "deposit_liability": round(state.deposit_liability, 2),
            "rental_revenue": state.cumulative_rental_revenue,
            "retail_revenue": state.cumulative_retail_revenue,
            "purchase_cost": state.cumulative_purchase_cost,
            **(
                {
                    "supplier_trial_handling_fees": (
                        state.cumulative_supplier_trial_handling_fees
                    ),
                    "supplier_trial_usage_fees": (
                        state.cumulative_supplier_trial_usage_fees
                    ),
                    "supplier_trial": self.environment._supplier_trial_summary(
                        state
                    ),
                }
                if self.environment._v37_supplier_trial()
                else {}
            ),
            **(
                {
                    "reservation_contingency_inquiries": (
                        state.cumulative_reservation_contingency_inquiries
                    ),
                    "reservation_contingencies_applied": (
                        state.cumulative_reservation_contingencies_applied
                    ),
                    "reservation_contingency_customer_cancellations": (
                        state.cumulative_reservation_contingency_customer_cancellations
                    ),
                }
                if self.environment._v38_proactive_contingency()
                else {}
            ),
            "cleaning_cost": state.cumulative_cleaning_cost,
            "repair_cost": state.cumulative_repair_cost,
            "asset_sale_proceeds": state.cumulative_asset_sale_proceeds,
            "assets_sold": state.cumulative_assets_sold,
            "retail_liquidation_proceeds": (
                state.cumulative_retail_liquidation_proceeds
            ),
            "retail_liquidation_book_cost": (
                state.cumulative_retail_liquidation_book_cost
            ),
            "retail_liquidation_loss": state.cumulative_retail_liquidation_loss,
            "retail_units_liquidated": state.cumulative_retail_units_liquidated,
            "retail_holding_cost": state.cumulative_retail_holding_cost,
            "rental_holding_cost": state.cumulative_rental_holding_cost,
            "operating_fees": state.cumulative_fees,
            "model_cost": state.cumulative_model_cost,
            "customers": state.cumulative_customers,
            "transactions": state.cumulative_transactions,
            "full_baskets": state.cumulative_full_baskets,
            "partial_baskets": state.cumulative_partial_baskets,
            "unfilled_baskets": state.cumulative_unfilled_baskets,
            "operationally_evaluable_baskets": (
                state.cumulative_operationally_evaluable_baskets
            ),
            "operational_full_baskets": state.cumulative_operational_full_baskets,
            "operational_partial_baskets": state.cumulative_operational_partial_baskets,
            "operational_unfilled_baskets": state.cumulative_operational_unfilled_baskets,
            "requested_units": state.cumulative_requested_units,
            "fulfilled_units": state.cumulative_fulfilled_units,
            "outside_rejections": state.cumulative_outside_rejections,
            "stockout_rejections": state.cumulative_stockout_rejections,
            "service_closed_rejections": state.cumulative_service_closed_rejections,
            "capacity_policy_rejections": int(
                getattr(state, "cumulative_capacity_policy_rejections", 0)
            ),
            "horizon_rejections": state.cumulative_horizon_rejections,
            "unpriced_rejections": int(
                getattr(state, "cumulative_unpriced_rejections", 0)
            ),
            "reservation_compensation": round(
                float(state.cumulative_reservation_compensation), 2
            ),
            "reservations": {
                "requests": int(state.cumulative_reservation_requests),
                "quotes": int(state.cumulative_reservation_quotes),
                "confirmed": int(state.cumulative_reservations_confirmed),
                "pickups": int(state.cumulative_reservation_pickups),
                "pickup_failures": int(
                    state.cumulative_reservation_pickup_failures
                ),
                "customer_cancellations": int(
                    state.cumulative_reservation_customer_cancellations
                ),
                **(
                    {
                        "weather_customer_cancellations": sum(
                            event.event_type == "customer_cancelled_weather"
                            for event in state.reservation_events
                        ),
                        "non_weather_customer_cancellations": sum(
                            event.event_type == "customer_cancelled_non_weather"
                            for event in state.reservation_events
                        ),
                        "weather_reviews_kept_original": sum(
                            event.event_type == "weather_review_no_change"
                            for event in state.reservation_events
                        ),
                    }
                    if self.environment._v42_weather_information_coherence()
                    else {}
                ),
                "manager_cancellations": int(manager_cancellations),
                **(
                    {
                        "proactive_manager_cancellations": sum(
                            event.details.get("cancellation_origin")
                            == "proactive"
                            for event in manager_cancellation_events
                        ),
                        "risk_alert_manager_cancellations": sum(
                            event.details.get("cancellation_origin")
                            == "risk_alert"
                            for event in manager_cancellation_events
                        ),
                        "same_day_manager_cancellations": sum(
                            int(event.details.get("notice_days", 0)) == 0
                            for event in manager_cancellation_events
                        ),
                        "manager_failure_required_service_unit_days": sum(
                            int(
                                event.details.get(
                                    "required_service_unit_days", 0
                                )
                            )
                            for event in state.reservation_events
                            if event.event_type
                            in {"cancelled_by_manager", "pickup_failed"}
                        ),
                        "service_reputation": (
                            self.environment._reservation_service_reputation(
                                state
                            )
                        ),
                    }
                    if v29_summary
                    else {}
                ),
            },
            "eligible_requested_units": max(
                0, state.cumulative_requested_units - state.cumulative_horizon_rejections
            ),
            "model_calls": self.store.model_call_count(),
            "forced_day_ends": state.forced_day_ends,
            "parallel_tool_call_anomalies": state.parallel_tool_call_anomalies,
            "discarded_tool_calls": state.discarded_tool_calls,
            "read_only_tool_batches": state.read_only_tool_batches,
            "batched_read_only_tool_calls": state.batched_read_only_tool_calls,
            "rejected_tool_batches": state.rejected_tool_batches,
            "rejected_batched_tool_calls": state.rejected_batched_tool_calls,
        }
