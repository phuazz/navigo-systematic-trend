"""Phase-1 valuation layer: reconciliation gate + the three silent-failure guards.

The reconciliation gate is the whole point of Phase 1 — it proves Navigo's own
mark reproduces the engine's daily ``live_equity`` from the same closes, so that
when the engine's daily job is eventually retired (Phase 3) the headline NAV does
not move. The fixture is a frozen snapshot (engine live_track + yfinance closes/FX
on the capture date); the test never touches the network.
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

import valuation
from config import load_registry

REG = load_registry("navigo-systematic-trend")
FIXTURE = Path(__file__).parent / "fixtures" / "valuation_recon_2026-06-25.json"

# Observed within-week noise floor on the settled days is <= 0.4 bps (identical
# closes, identical FX convention, no cost). 2 bps gives rounding/FX-timing
# headroom without being tuned to force a pass: the settled days clear it ~5x.
SETTLED_TOL_BPS = 2.0
# The engine's run-day (freshest) bar is the least-settled close and can differ
# from our later fetch by ordinary last-bar/timing noise; this is a gross-error
# tripwire on that one point, not a tolerance the settled gate relies on.
RUNDAY_TRIPWIRE_BPS = 75.0


def _series(mapping: dict) -> pd.Series:
    idx = pd.to_datetime(list(mapping.keys()))
    return pd.Series(list(mapping.values()), index=idx, dtype="float64").sort_index()


def _load_fixture():
    fx_raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    live = fx_raw["live_track"]
    closes = {tk: _series(s) for tk, s in fx_raw["closes"].items()}
    fx = {"EUR": _series(fx_raw["fx"]["EURUSD"]), "CNY": _series(fx_raw["fx"]["CNYUSD"])}
    return live, closes, fx


# --- the reconciliation gate ------------------------------------------------
def test_reconciles_to_engine_live_equity():
    live, closes, fx = _load_fixture()
    mark = valuation.mark_to_market(live, closes, fx, REG)

    # Coverage must be complete — an uncovered weight understates the mark and
    # would make any reconciliation meaningless.
    assert mark["coverage"]["complete"], mark["coverage"]
    assert abs(mark["coverage"]["covered_weight"] - 1.0) < 1e-4

    rec = valuation.reconcile(mark, live)

    # Human-readable deviation table (visible under `pytest -s`).
    print("\n  reconciliation vs engine live_equity (anchor "
          f"{mark['anchor_date']} @ {mark['anchor_equity']}):")
    print(f"  {'date':12s}{'engine':>11s}{'navigo':>11s}{'dev_bps':>10s}  seg")
    for r in rec["per_date"]:
        print(f"  {r['date']:12s}{r['engine']:11.6f}{r['navigo']:11.6f}"
              f"{r['dev_bps']:10.2f}  {'settled' if r['settled'] else 'run-day'}")
    print(f"  settled: max {rec['settled']['max_abs_bps']} bps, mean "
          f"{rec['settled']['mean_abs_bps']} bps (n={rec['settled']['n']}); "
          f"incl run-day: max {rec['all']['max_abs_bps']} bps")

    # The gate: every settled within-week day ties to the engine within noise.
    assert rec["settled"]["n"] >= 5
    assert rec["settled"]["max_abs_bps"] <= SETTLED_TOL_BPS
    # The run-day bar is bounded only by the gross-error tripwire.
    assert rec["all"]["max_abs_bps"] <= RUNDAY_TRIPWIRE_BPS


def test_fx_conversion_is_necessary_and_correct():
    """Guard #1: marking non-base holdings without FX conversion drifts. A
    constant 1.0 FX (i.e. treat EUR/CNY closes as if USD) must reconcile materially
    worse than the real FX path — proving the conversion is both applied and right.
    """
    live, closes, fx = _load_fixture()
    fx_adj = valuation.reconcile(valuation.mark_to_market(live, closes, fx, REG), live)

    flat = {c: pd.Series(1.0, index=s.index) for c, s in fx.items()}
    no_fx = valuation.reconcile(valuation.mark_to_market(live, closes, flat, REG), live)

    assert fx_adj["settled"]["max_abs_bps"] <= SETTLED_TOL_BPS
    assert no_fx["settled"]["max_abs_bps"] > 3 * fx_adj["settled"]["max_abs_bps"]
    assert no_fx["settled"]["max_abs_bps"] > 5.0


# --- pure-logic check (synthetic, exact) ------------------------------------
def test_mark_arithmetic_exact():
    live = {"anchor_date": "2026-03-02", "anchor_equity": 100.0,
            "effective_weights": {"AAA": 0.6, "BBB.DE": 0.4}}
    closes = {"AAA": _series({"2026-03-02": 100.0, "2026-03-03": 110.0}),
              "BBB.DE": _series({"2026-03-02": 50.0, "2026-03-03": 55.0})}
    fx = {"EUR": _series({"2026-03-02": 1.10, "2026-03-03": 1.20})}
    # AAA (USD): 110/100 = 1.10. BBB (EUR->USD): (55*1.20)/(50*1.10) = 66/55 = 1.20.
    # NAV = 100 * (0.6*1.10 + 0.4*1.20) = 100 * 1.14 = 114.0.
    reg = {"base_currency": "USD",
           "etf_meta": {"AAA": {"tradeAs": "AAA"}, "BBB.DE": {"tradeAs": "BBB.DE"}}}
    mark = valuation.mark_to_market(live, closes, fx, reg)
    assert mark["dates"] == ["2026-03-03"]
    assert mark["equity"] == [114.0]
    assert mark["weights_as_of"] == "2026-03-02" and mark["nav_as_of"] == "2026-03-03"
    assert mark["coverage"]["complete"]


def test_two_as_of_stamps_never_collapse():
    """Guard #2: weights_as_of and nav_as_of are distinct fields and the gap is
    reported, so a stale-weights mark can never read as a single current date."""
    live, closes, fx = _load_fixture()
    mark = valuation.mark_to_market(live, closes, fx, REG)
    assert mark["weights_as_of"] == live["anchor_date"]
    assert mark["nav_as_of"] > mark["weights_as_of"]
    assert mark["weights_age_bdays"] == int(np.busday_count(
        dt.date.fromisoformat(mark["weights_as_of"]),
        dt.date.fromisoformat(mark["nav_as_of"])))
    assert mark["weights_age_bdays"] >= 1


# --- look-ahead guard (#3) --------------------------------------------------
def test_marks_only_strictly_forward_and_respects_asof():
    live, closes, fx = _load_fixture()
    anchor = live["anchor_date"]
    mark = valuation.mark_to_market(live, closes, fx, REG)
    assert all(d > anchor for d in mark["dates"])      # never the anchor or earlier
    assert mark["dates"] == sorted(mark["dates"])      # monotone, no look-ahead jumble

    # asof caps the horizon — a date with an available close beyond asof is excluded.
    capped = valuation.mark_to_market(live, closes, fx, REG, asof=live["live_dates"][2])
    assert capped["nav_as_of"] == live["live_dates"][2]
    assert all(d <= live["live_dates"][2] for d in capped["dates"])


def test_uncovered_weight_is_flagged_not_hidden():
    """A missing price must surface, never be silently renormalised away."""
    live = {"anchor_date": "2026-03-02", "anchor_equity": 100.0,
            "effective_weights": {"AAA": 0.6, "ZZZ": 0.4}}
    closes = {"AAA": _series({"2026-03-02": 100.0, "2026-03-03": 110.0})}  # ZZZ absent
    reg = {"base_currency": "USD", "etf_meta": {"AAA": {"tradeAs": "AAA"}}}
    mark = valuation.mark_to_market(live, closes, {}, reg)
    assert mark["coverage"]["uncovered"] == ["ZZZ"]
    assert not mark["coverage"]["complete"]
    assert abs(mark["coverage"]["covered_weight"] - 0.6) < 1e-9


# --- date-boundary tests (CLAUDE.md: one month, one year) -------------------
def _one_asset(anchor, anchor_equity, close_map):
    live = {"anchor_date": anchor, "anchor_equity": anchor_equity,
            "effective_weights": {"ZZZ": 1.0}}
    closes = {"ZZZ": _series(close_map)}
    reg = {"base_currency": "USD", "etf_meta": {"ZZZ": {"tradeAs": "ZZZ"}}}
    return valuation.mark_to_market(live, closes, {}, reg)


def test_month_boundary():
    # Anchor on the last business day of January; mark across into February.
    mark = _one_asset("2025-01-31", 1.0,
                      {"2025-01-31": 100.0, "2025-02-03": 101.0, "2025-02-04": 102.0})
    assert "2025-01-31" not in mark["dates"]            # anchor excluded
    assert mark["dates"] == ["2025-02-03", "2025-02-04"]
    assert mark["equity"] == [1.01, 1.02]
    assert mark["nav_as_of"] == "2025-02-04"
    assert mark["weights_age_bdays"] == int(
        np.busday_count(dt.date(2025, 1, 31), dt.date(2025, 2, 4)))  # == 2


def test_year_boundary():
    # Anchor on the last business day of 2024; mark across into 2025.
    mark = _one_asset("2024-12-31", 1.0,
                      {"2024-12-31": 200.0, "2025-01-02": 206.0, "2025-01-03": 210.0})
    assert "2024-12-31" not in mark["dates"]
    assert mark["dates"] == ["2025-01-02", "2025-01-03"]
    assert mark["equity"] == [1.03, 1.05]
    assert mark["nav_as_of"] == "2025-01-03"
    assert mark["weights_age_bdays"] == int(
        np.busday_count(dt.date(2024, 12, 31), dt.date(2025, 1, 3)))
