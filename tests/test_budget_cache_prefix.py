"""Dynamic budget notices must not invalidate unchanged conversation prefixes."""
import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from outfitter.config import OutfitterConfig
from outfitter.llm import ScriptedProvider
from outfitter.runner import OutfitterRunner


def config(tmp_path, **overrides):
    # Same tests run in the research tree and the portable release checkout.
    for root in Path(__file__).resolve().parents:
        for name in ("outfitter_v54_main.example.yaml", "outfitter_v54_main.yaml"):
            candidate = root / "configs" / name
            if candidate.is_file():
                return replace(OutfitterConfig.from_yaml(candidate), runs_dir=str(tmp_path), **overrides)
    raise AssertionError("V53 configuration not found")


def call(name, identifier):
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": identifier, "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }]}


def budget(request):
    last = request["messages"][-1]
    assert last["role"] == "user"
    assert last["content"].startswith("Workspace budget: ")
    return json.loads(last["content"].split(": ", 1)[1])


def test_requests_append_history_and_exact_budget_notices(tmp_path):
    provider = ScriptedProvider([call("business_status", "status"),
        call("business_status", "status-2"), call("close_business_day", "close")])
    runner = OutfitterRunner.create(config=config(tmp_path), provider=provider, run_id="prefix")
    summary = runner.run(stop_after_day=1)
    assert summary["day"] == 2 and not summary["done"]
    requests = provider.requests
    assert [budget(r)["remaining_responses_today_including_next"] for r in requests] == [200, 199, 198]
    for before, after in zip(requests, requests[1:]):
        assert after["messages"][:len(before["messages"])] == before["messages"]
        assert after["tools"] == before["tools"]
    persisted = runner.store.context_messages_for_day(1)
    notices = [m for m in persisted if str(m.get("content", "")).startswith("Workspace budget: ")]
    assert notices == [r["messages"][-1] for r in requests]
    for index, message in enumerate(persisted):
        if message in notices:
            assert persisted[index + 1]["role"] == "assistant"
    manifest = json.loads((runner.store.run_dir / "manifest.json").read_text())
    assert manifest["controller"]["workspace_budget_message_policy"] == "append-only-tail-v1"


def test_midday_resume_keeps_prefix_and_fresh_day_drops_old_notices(tmp_path):
    first = ScriptedProvider([call("business_status", "status")])
    runner = OutfitterRunner.create(config=config(tmp_path), provider=first, run_id="resume-prefix")
    runner.run(stop_after_model_calls=1)
    second = ScriptedProvider([call("close_business_day", "close-1"), call("close_business_day", "close-2")])
    resumed = OutfitterRunner.resume(run_dir=runner.store.run_dir, provider=second)
    resumed.run(stop_after_day=2)
    original = first.requests[0]["messages"]
    assert second.requests[0]["messages"][:len(original)] == original
    assert budget(second.requests[0])["remaining_responses_today_including_next"] == 199
    assert budget(second.requests[1])["day"] == 2
    assert budget(second.requests[1])["remaining_responses_today_including_next"] == 200
    assert budget(second.requests[1])["remaining_responses_in_run_including_next"] == 39998
    assert not any(m["role"] == "assistant" for m in second.requests[1]["messages"])
    assert sum(str(m.get("content", "")).startswith("Workspace budget: ")
               for m in second.requests[1]["messages"]) == 1


def test_api_failure_does_not_persist_or_duplicate_budget(tmp_path):
    class FailingProvider:
        def complete(self, **kwargs):
            self.request = copy.deepcopy(kwargs)
            raise RuntimeError("simulated API failure")

    provider = FailingProvider()
    runner = OutfitterRunner.create(config=config(tmp_path), provider=provider, run_id="retry-prefix")
    with pytest.raises(RuntimeError, match="simulated API failure"):
        runner.run(stop_after_day=1)
    assert runner.store.model_call_count() == 0
    assert not any(str(m.get("content", "")).startswith("Workspace budget: ")
                   for m in runner.store.context_messages_for_day(1))
    replacement = ScriptedProvider([call("close_business_day", "close")])
    resumed = OutfitterRunner.resume(run_dir=runner.store.run_dir, provider=replacement)
    resumed.run(stop_after_day=1)
    assert replacement.requests[0] == provider.request
    assert sum(str(m.get("content", "")).startswith("Workspace budget: ")
               for m in resumed.store.context_messages_for_day(1)) == 1


def test_notice_and_response_roll_back_together_on_store_failure(tmp_path):
    runner = OutfitterRunner.create(config=config(tmp_path), provider=ScriptedProvider([]),
        run_id="atomic-notice")
    state = runner.store.load_state()
    notice = runner._build_context(state)[-1]
    before = runner.store.messages()
    with pytest.raises(TypeError):
        runner.store.record_model_response(state, request={}, response={},
            assistant_message={"role": "assistant", "content": object()},
            input_tokens=0, output_tokens=0, charge=lambda *args: None,
            request_tail_message=notice)
    assert runner.store.messages() == before
    assert runner.store.model_call_count() == 0


def test_last_permitted_tool_executes_before_forced_close(tmp_path):
    provider = ScriptedProvider([call("business_status", "one"), call("business_status", "two")])
    runner = OutfitterRunner.create(config=config(tmp_path, max_iterations_per_day=2),
        provider=provider, run_id="last-response")
    runner.run(stop_after_day=1)
    assert [budget(r)["remaining_responses_today_including_next"] for r in provider.requests] == [2, 1]
    assert runner.store.tool_result_exists("two")
    assert runner.store.load_state().day == 2
    assert runner.store.load_state().forced_day_ends == 1


def test_context_build_is_read_only_and_budget_survives_truncation(tmp_path):
    runner = OutfitterRunner.create(config=config(tmp_path, context_budget_chars=20000),
        provider=ScriptedProvider([]), run_id="truncate")
    state = runner.store.load_state()
    runner.store.record_day_start(state, {"role": "user", "content": "morning"})
    for index in range(20):
        runner.store.append_message(1, "controller", {"role": "user", "content": str(index) + "x" * 2000})
    runner.store.append_message(1, "controller", call("business_status", "paired"))
    runner.store.append_message(1, "controller", {"role": "tool", "tool_call_id": "paired", "content": "{}"})
    before = runner.store.messages()
    context = runner._build_context(state)
    assert runner._build_context(state) == context
    assert runner.store.messages() == before
    assert budget({"messages": context})["remaining_responses_today_including_next"] == 200
    assert any("older messages from today" in str(m.get("content", "")) for m in context)
    for index, message in enumerate(context):
        if message["role"] == "tool":
            assert context[index - 1]["role"] in {"assistant", "tool"}
    assert context[-2]["role"] == "tool"
