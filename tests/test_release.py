from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from outfitter.actor_interface_slimming import POLICY_ID as V41_POLICY_ID
from outfitter.actor_manual import POLICY_ID as MANUAL_POLICY
from outfitter.recovery_state import ENVIRONMENT_PROTOCOL
from outfitter.config import OutfitterConfig
from outfitter.data import OutfitterDataset
from outfitter.environment import OutfitterEnvironment
from outfitter.prompts import system_prompt
from outfitter.run_scripted_merchant import run as run_scripted
from outfitter.tool_executor import ToolExecutor
from outfitter.weather_information import (
    POLICY_ID,
    consideration_probability,
)


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/outfitter_v54_main.example.yaml"
SECONDARY = ROOT / "configs/outfitter_v54_secondary.example.yaml"


def test_standalone_cli_does_not_require_research_modules() -> None:
    from outfitter.cli import build_parser

    assert "run" in build_parser().format_help()


def environment(path: Path = MAIN) -> OutfitterEnvironment:
    config = OutfitterConfig.from_yaml(path)
    return OutfitterEnvironment(config, OutfitterDataset.load(config.data_dir))


def integrity_paths(value, prefix: str = "$") -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            path = f"{prefix}.{key}"
            if normalized in {"hash", "sha256", "fingerprint"} or normalized.endswith(
                ("_hash", "_sha256", "_fingerprint")
            ):
                result.append(path)
            result.extend(integrity_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(integrity_paths(child, f"{prefix}[{index}]"))
    return result


def test_runtime_loads_from_the_installed_package() -> None:
    config = OutfitterConfig.from_yaml(MAIN)
    assert Path(config.data_dir).resolve() == Path(
        __import__("outfitter").__file__
    ).resolve().parent / "runtime_data"
    env = environment()
    state = env.reset()
    assert env.dataset.metadata["environment_protocol"] == ENVIRONMENT_PROTOCOL
    assert env.dataset.metadata["actor_interface_slimming_policy"] == V41_POLICY_ID
    assert env.dataset.metadata["weather_information_policy"] == POLICY_ID
    assert env._v40_revolving_durable_capital()
    assert env._v41_actor_interface_slimming()
    assert env._v42_weather_information_coherence()
    assert env.dataset.metadata["business_manual_policy"] == MANUAL_POLICY
    assert ToolExecutor(env).tool_interface_version == 52
    assert env._v50_summer_pressure()
    assert env._v48_service_guarantee()
    assert env._v47_autonomous_rentals() and not env._v46_return_dynamics()
    assert env._v45_reservation_difficulty()
    assert env.dataset.metadata["reservation_calendar_peak_multiplier"] == 1.4
    assert config.durable_acquisition_budget == 1000.0
    assert state.day == 1


def test_tracks_differ_only_in_assortment_mode() -> None:
    main = OutfitterConfig.from_yaml(MAIN).to_dict()
    secondary = OutfitterConfig.from_yaml(SECONDARY).to_dict()
    assert main.pop("assortment_mode") == "rental_only"
    assert secondary.pop("assortment_mode") == "rental_and_retail"
    assert main == secondary


def test_actor_surfaces_do_not_expose_integrity_hashes() -> None:
    env = environment()
    state = env.reset()
    tools = ToolExecutor(env)
    for payload in (
        env.public_operating_rules(),
        env.observation(state, consume_inbox=False),
        tools.execute(state, "business_status", {}).data,
        tools.definitions(),
    ):
        assert integrity_paths(payload) == []
    assert state.warehouse_capacity["weights_sha256"]
    assert env.dataset.metadata["content_sha256"]


def test_short_scripted_run_uses_no_model(tmp_path: Path) -> None:
    config = replace(
        OutfitterConfig.from_yaml(MAIN),
        runs_dir=str(tmp_path),
        max_days=2,
        runoff_days=2,
    )
    result = run_scripted(
        config,
        "calendar",
        run_id="portable-no-llm-smoke",
    )
    assert result["done"] is True
    assert result["scripted_merchant"]["model_calls"] == 0
    assert result["scripted_merchant"]["tool_error_records"] == []


def test_release_has_no_vending_bench_package_or_import() -> None:
    assert not (ROOT / "src/vending_bench").exists()
    for path in (ROOT / "src/outfitter").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "from vending_bench" not in text
        assert "import vending_bench" not in text


def test_reservation_weather_consideration_uses_public_forecast_response() -> None:
    assert consideration_probability(
        archetype="overnight", regime="rainy", forecast_trust=1.0
    ) < consideration_probability(
        archetype="overnight", regime="rainy", forecast_trust=0.0
    )


def test_system_prompt_has_public_rules_not_customer_algorithm() -> None:
    env = environment()
    rules = env.public_operating_rules()
    tools = ToolExecutor(env)
    prompt = system_prompt(
        rules,
        tool_call_protocol="read_only_batch_v1",
        read_only_tools=tools.read_only_names(),
        max_read_only_batch_size=env.config.max_read_only_batch_size,
    )
    assert "read_business_rules" in prompt
    assert "200 model responses" in prompt
    assert "one fixed channel opportunity" not in prompt
    assert "independent non-weather customer event" not in prompt
    # Detailed billing/query instructions belong in the handbook/tool descriptions.
    assert "55% nominal recovery baseline" in prompt
    assert "fresh model session" in prompt
    assert "may create a binding commitment immediately or at close" in prompt
    assert len(prompt) <= 5000


def test_rule_tool_is_read_only_and_supports_topics() -> None:
    env = environment()
    state = env.reset()
    before = state.to_dict()
    tools = ToolExecutor(env)
    assert tools.is_read_only("read_business_rules")
    directory = tools.execute(state, "read_business_rules", {})
    assert directory.ok and len(directory.data["topics"]) == 6
    chapters = tools.execute(state, "read_business_rules", {"topics": ["finance", "reservations"]})
    assert chapters.ok and len(chapters.data["chapters"]) == 2
    assert state.to_dict() == before


def test_runner_freezes_and_checks_rulebook(tmp_path: Path) -> None:
    import json
    import pytest
    from outfitter.llm import ScriptedProvider
    from outfitter.runner import OutfitterRunner

    config = replace(environment().config, runs_dir=str(tmp_path), max_days=2, runoff_days=0)
    runner = OutfitterRunner.create(config=config, provider=ScriptedProvider([]), run_id="rules")
    manifest = json.loads((runner.store.run_dir / "manifest.json").read_text())
    assert manifest["controller"]["business_manual_policy"] == MANUAL_POLICY
    path = runner.store.run_dir / "business_rules.json"
    resumed = OutfitterRunner.resume(run_dir=runner.store.run_dir, provider=ScriptedProvider([]))
    assert resumed.tools.business_rules_snapshot == runner.tools.business_rules_snapshot
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="integrity"):
        OutfitterRunner.resume(run_dir=runner.store.run_dir, provider=ScriptedProvider([]))
