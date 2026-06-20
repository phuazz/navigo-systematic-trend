"""Adapter logic that is easy to get subtly wrong: the weight build-decomposition
(it must reproduce the engine's effective weights from sleeve allocation x within-
sleeve weight, plus the EM tilt) and the exposure roll-ups.
"""
import adapter
from config import load_registry

REG = load_registry("navigo-systematic-trend")


def _live(tilt=True):
    return {
        "computed_at_utc": "2026-06-19T22:50:00Z",
        "eem_tilt_active": tilt,
        "regime_state": "RISK_ON",
        "anchor_date": "2026-06-17", "anchor_equity": 2.9638,
        "live_dates": ["2026-06-18"], "live_equity": [3.0052],
        # Effective weights consistent with the sleeve weights below + 10% tilt.
        "effective_weights": {"EEM": 0.14958, "SOXX": 0.13583, "EXH1": 0.07214, "SHY": 0.00003},
        "sleeve_extensions": {
            "strategy_a": {"weights": {"SOXX": 0.3881}},
            "strategy_b": {"weights": {"EEM": 0.1983}},
            "strategy_c": {"weights": {}},
            "strategy_d": {"weights": {"EXH1": 0.3607}},
        },
    }


def test_eem_build_combines_sleeve_b_and_tilt():
    w = adapter.build_weights(_live(tilt=True), REG)
    eem = next(r for r in w["rows"] if r["ticker"] == "EEM")
    sleeves = {b["sleeve"] for b in eem["build"]}
    assert sleeves == {"B", "TILT"}
    total = sum(b["contrib"] for b in eem["build"])
    assert abs(total - eem["weight"]) < 1e-3          # reconstruct the effective weight
    # Sleeve B allocation is reduced to 25% while the tilt is on.
    b_leg = next(b for b in eem["build"] if b["sleeve"] == "B")
    assert abs(b_leg["alloc"] - 0.25) < 1e-9


def test_sector_holding_single_sleeve_leg():
    w = adapter.build_weights(_live(), REG)
    soxx = next(r for r in w["rows"] if r["ticker"] == "SOXX")
    assert [b["sleeve"] for b in soxx["build"]] == ["A"]
    assert abs(soxx["build"][0]["contrib"] - 0.35 * 0.3881) < 1e-4


def test_exposure_rollups_and_concentration():
    w = adapter.build_weights(_live(), REG)
    # Europe oil & gas should land in sleeve D and Europe geography.
    assert "D" in w["by_sleeve"] and "Europe" in w["by_geo"]
    assert w["concentration"]["n_holdings"] >= 3
    assert 0 < w["concentration"]["hhi"] <= 1


def test_tilt_off_drops_tilt_leg_and_restores_b_alloc():
    w = adapter.build_weights(_live(tilt=False), REG)
    eem = next(r for r in w["rows"] if r["ticker"] == "EEM")
    assert all(b["sleeve"] != "TILT" for b in eem["build"])
    b_leg = next(b for b in eem["build"] if b["sleeve"] == "B")
    assert abs(b_leg["alloc"] - 0.35) < 1e-9           # full 35% when tilt is off
