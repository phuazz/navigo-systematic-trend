"""Statistics and — per the vault date rules — explicit date-boundary tests
(one month boundary, one year boundary). All anchoring is asserted against a
direct computation so a regression in the anchor logic fails loudly.
"""
import numpy as np
import pandas as pd

import metrics


def ramp(start, end, daily=0.001):
    """Business-day growth-of-1 series compounding `daily` each day."""
    idx = pd.bdate_range(start, end)
    return pd.Series((1 + daily) ** np.arange(len(idx)), index=idx, dtype="float64")


def test_cagr_uses_actual_calendar_span():
    eq = metrics.equity_series(["2020-01-01", "2021-01-01"], [1.0, 2.0])
    # Doubling over ~1 calendar year -> ~100% CAGR.
    assert abs(metrics.cagr(eq) - 1.0) < 0.02


def test_drawdown_and_sharpe_signs():
    eq = metrics.equity_series(["2020-01-01", "2020-01-02", "2020-01-03"], [1.0, 0.9, 1.1])
    assert metrics.max_drawdown(eq) <= -0.09          # dipped 10%
    up = ramp("2021-01-01", "2021-06-01", daily=0.001)
    assert metrics.sharpe(up) > 0                      # monotone up -> positive Sharpe


def test_period_returns_year_boundary():
    # Spans the 2025->2026 year boundary; YTD must anchor on the last 2025 close.
    eq = ramp("2025-12-15", "2026-01-15")
    pr = metrics.period_returns(eq)
    anchor = eq[eq.index <= pd.Timestamp("2025-12-31")].iloc[-1]
    assert abs(pr["YTD"] - (eq.iloc[-1] / anchor - 1.0)) < 1e-9
    # The 2025 year-end was a Wednesday; sanity-check we picked it.
    assert eq[eq.index <= pd.Timestamp("2025-12-31")].index[-1] == pd.Timestamp("2025-12-31")


def test_period_returns_month_boundary():
    # Spans Jan->Feb 2026; MTD must anchor on the last January close.
    eq = ramp("2026-01-20", "2026-02-10")
    pr = metrics.period_returns(eq)
    anchor = eq[eq.index <= pd.Timestamp("2026-01-31")].iloc[-1]
    assert abs(pr["MTD"] - (eq.iloc[-1] / anchor - 1.0)) < 1e-9


def test_monthly_matrix_has_year_and_months():
    eq = ramp("2024-11-01", "2025-02-28")
    mm = metrics.monthly_matrix(eq)
    assert 2024 in mm and 2025 in mm
    assert "YEAR" in mm[2024] and 12 in mm[2024]      # month is 1-indexed
    # YEAR compounding equals product of monthly returns within the year.
    prod = 1.0
    for mo in range(1, 13):
        if mo in mm[2025]:
            prod *= 1 + mm[2025][mo]
    assert abs(mm[2025]["YEAR"] - (prod - 1.0)) < 1e-9


def test_capture_ratios_directional():
    bench = ramp("2022-01-01", "2024-01-01", daily=0.0005)
    model = ramp("2022-01-01", "2024-01-01", daily=0.0003)  # tracks at lower slope
    caps = metrics.capture_ratios(model, bench)
    assert 0 < caps["up_capture"] < 1.5
    assert caps["beta"] is not None
