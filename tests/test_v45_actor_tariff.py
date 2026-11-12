"""Exercise the rendered actor contract, including the final tool-description layer."""
from pathlib import Path
import json

import pytest

from outfitter.config import OutfitterConfig
from outfitter.data import OutfitterDataset
from outfitter.environment import OutfitterEnvironment
from outfitter.prompts import system_prompt
from outfitter.reservation_difficulty import LINE_PRICING_BASIS, ASSET_PRICING_BASIS, rental_amount
from outfitter.tool_executor import ToolExecutor


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/outfitter_v54_main.example.yaml"
SECONDARY = ROOT / "configs/outfitter_v54_secondary.example.yaml"


def environment(path):
    config = OutfitterConfig.from_yaml(path)
    return OutfitterEnvironment(config, OutfitterDataset.load(config.data_dir))


@pytest.mark.parametrize("path", [MAIN, SECONDARY])
def test_actual_prompt_handbook_and_tools_agree_on_tariff(path):
    env = environment(path)
    tools = ToolExecutor(env)
    prompt = system_prompt(env.public_operating_rules(), tool_call_protocol="first_only",
                           read_only_tools=tools.read_only_names(), max_read_only_batch_size=5)
    rules = json.dumps(env.business_manual())
    assert "Walk-ins use posted prices" in prompt
    assert "Pricing units" in prompt and "described by its tools and the handbook" in prompt
    assert "first-day prices per asset" in rules
    assert "per asset per day" not in prompt
    assert "whole-line daily rent" not in prompt
    assert "per-asset daily prices" not in rules
    assert "round_half_up_to_cents(0.5 * P)" in rules
    assert "once per commitment version" in rules
    definitions = {row["function"]["name"]: row["function"] for row in tools.definitions()}
    assert "first-day" in definitions["set_prices"]["description"]
    assert "per asset-day" not in definitions["set_prices"]["description"]
    assert "per whole line" in definitions["respond_reservation_service_offers"]["description"]
    quote_schema = json.dumps(definitions["quote_reservation_inquiries"]["parameters"])
    assert "first-day rental price per asset" in quote_schema
    recovery = definitions["resolve_reservation_risks"]
    assert "inquire_partial_fulfillment" in recovery["description"]
    assert "select_partial_fulfillment" in recovery["description"]


@pytest.mark.parametrize("path", [MAIN, SECONDARY])
def test_fixed_multiday_pricing_survives_workspace_confirmation_and_detail(path):
    env = environment(path)
    state = env.reset()
    tools = ToolExecutor(env)
    request = None
    for _ in range(14):
        request = next((row for row in state.reservation_requests
                        if row.status == "pending_service_offer"
                        and row.return_day - row.pickup_day > 1), None)
        if request is not None:
            break
        env.close_business_day(state)
    assert request is not None, "fixture needs a multi-day fixed offer"
    before = state.to_dict()
    workspace = tools.execute(state, "view_reservation_quote_workspace", {
        "mode": "requests", "request_ids": [request.request_id],
        "offer_projection": "references_only",
    })
    assert workspace.ok
    assert state.to_dict() == before
    assert workspace.data["quote_constraints"]["pricing_basis"] == ASSET_PRICING_BASIS
    offer = workspace.data["requests"][0]["service_offer_terms"]
    assert offer["pricing_basis"] == LINE_PRICING_BASIS
    plan = offer["complete_plans"][0]
    for line in plan["lines"]:
        assert line["pricing_basis"] == LINE_PRICING_BASIS
        assert line["line_rent"] == rental_amount(
            line["line_daily_rent"], request.return_day - request.pickup_day)
    confirmation = tools.execute(state, "respond_reservation_service_offers", {"decisions": [{
        "request_id": request.request_id, "action": "accept", "plan_id": plan["plan_id"],
        "pickup_day": request.pickup_date_options[0]["pickup_day"],
    }]})
    assert confirmation.ok
    result = confirmation.data["decisions"][0]
    assert result["total_locked_rent"] == offer["total_rent"]
    assert all(line["pricing_basis"] == LINE_PRICING_BASIS for line in result["lines"])
    detail = tools.execute(state, "view_reservation", {
        "view": "detail", "identifiers": [result["commitment_id"]],
        "sections": ["current_terms"],
    })
    assert detail.ok
    lines = detail.data["records"][0]["current_terms"]["locked_lines"]
    assert all(line["pricing_basis"] == LINE_PRICING_BASIS for line in lines)
    assert '"line_day"' not in json.dumps(detail.data)
