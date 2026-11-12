from __future__ import annotations

from pathlib import Path

import pytest

from outfitter.rental_service_comparison import build_comparison, render_figure


def _report(*, label="model", seed=42, status="in_progress", mismatch=0.0):
    daily = []
    official = 1000.0
    cumulative_managed = 0.0
    for day in (1, 2):
        # Rental: 10 + 1 - 2 - 1 - .5 - 1 - .5 - 1 - 1 = 4.
        # Retail: 5 - 2 - 1 = 2. Shared overhead: -1. Liquidation: -.5.
        # Operating P&L is 5; managed contribution after liquidation is 4.5.
        cumulative_managed += 4.5
        daily.append({
            "day": day,
            "calendar": {"holiday_name": "暑假" if day == 2 else None},
            "rental_revenue": 10.0,
            "damage_deductions": 1.0,
            "cleaning_cost": 2.0,
            "repair_cost": 1.0,
            "maintenance_cost": 0.5,
            "rental_holding_cost": 1.0,
            "rental_idle_holding_cost": 0.5,
            "rental_use_depreciation": 1.0,
            "reservation_compensation": 1.0,
            "retail_revenue": 5.0,
            "retail_cogs": 2.0,
            "retail_holding_cost": 1.0,
            "operating_fee": 1.0,
            "retail_liquidation_loss": 0.5,
            "daily_operating_pnl": 5.0 + mismatch,
            "official_liquidation_value_end": official + cumulative_managed - 3 * day,
            # These must not enter the service curve.
            "purchase_cost": 9999.0,
            "asset_sale_proceeds": 8888.0,
            "contract_adjusted_economic_value_end": -999999.0,
        })
    return {
        "run": {
            "run_id": f"{label}-{seed}",
            "model": label,
            "seed": seed,
            "max_days": 1,
            "environment_rules": {"frozen": True},
        },
        "headline": {
            "status": status,
            "initial_official_liquidation_value": official,
            "decision_horizon_day": 1,
            "runoff_end_day": 2,
        },
        "daily": daily,
    }


def test_curve_isolates_rental_service_and_reconciles_endpoint():
    result = build_comparison([("Model A", _report())], allow_in_progress=True)
    rows = result["curves"]
    assert [row["rental_service_contribution"] for row in rows] == [0.0, 4.0, 8.0]
    endpoint = result["endpoints"][0]
    assert endpoint["retail_merchandise_contribution_mean"] == 4.0
    assert endpoint["shared_operating_overhead_mean"] == -2.0
    assert endpoint["retail_liquidation_effect_mean"] == -1.0
    assert endpoint["managed_operating_contribution_mean"] == 9.0
    assert endpoint["official_value_change_mean"] == 3.0
    assert endpoint["capital_obligation_reconciliation_mean"] == -6.0
    assert result["metric"]["causal_effect_claimed"] is False
    assert result["metric"]["contract_adjusted_value_used"] is False


def test_daily_accounting_mismatch_fails_loudly():
    with pytest.raises(ValueError, match="does not reconcile"):
        build_comparison(
            [("Model A", _report(mismatch=1.0))], allow_in_progress=True
        )


@pytest.mark.parametrize("handling,usage", [(10.0, 0.0), (0.0, 2.75), (10.0, 2.75), (0.0, 0.0)])
def test_supplier_trial_fees_are_rental_costs_and_reconcile(handling, usage):
    report = _report()
    # A handling charge on day one, usage on day two. Neither is a
    # purchase principal, shared overhead, or capital reconciliation item.
    report["daily"][0]["supplier_trial_handling_fee"] = handling
    report["daily"][1]["supplier_trial_usage_fee"] = usage
    report["daily"][0]["daily_operating_pnl"] -= handling
    report["daily"][1]["daily_operating_pnl"] -= usage
    report["daily"][0]["official_liquidation_value_end"] -= handling
    report["daily"][1]["official_liquidation_value_end"] -= handling + usage
    result = build_comparison([("Model A", report)], allow_in_progress=True)
    rows = result["curves"]
    assert [row["rental_service_contribution"] for row in rows] == pytest.approx(
        [0.0, 4.0 - handling, 8.0 - handling - usage])
    endpoint = result["endpoints"][0]
    assert endpoint["managed_operating_contribution_mean"] == pytest.approx(9.0 - handling - usage)
    assert endpoint["shared_operating_overhead_mean"] == -2.0
    assert endpoint["capital_obligation_reconciliation_mean"] == -6.0
    assert {"supplier_trial_handling_fee", "supplier_trial_usage_fee"} <= set(result["metric"]["included_costs"])


def test_trial_cost_fix_does_not_mask_other_accounting_mismatch():
    report = _report()
    report["daily"][0]["supplier_trial_usage_fee"] = 2.0
    report["daily"][0]["daily_operating_pnl"] -= 3.0
    with pytest.raises(ValueError, match="does not reconcile"):
        build_comparison([("Model A", report)], allow_in_progress=True)


def test_in_progress_requires_opt_in_and_png_renders(tmp_path: Path):
    with pytest.raises(ValueError, match="allow-in-progress"):
        build_comparison([("Model A", _report())])
    result = build_comparison(
        [("Model A", _report(status="completed"))]
    )
    target = tmp_path / "rental.png"
    render_figure(result, target, dpi=72)
    assert target.is_file() and target.stat().st_size > 0


def test_multi_model_requires_matching_seed_sets():
    with pytest.raises(ValueError, match="same seed set"):
        build_comparison([
            ("A", _report(label="a", seed=42, status="completed")),
            ("B", _report(label="b", seed=73, status="completed")),
        ])
