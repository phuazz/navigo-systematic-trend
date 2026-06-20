"""Presentation-grade return/risk statistics computed from equity curves.

Conventions (stated once, applied everywhere):
  - Equity series are growth-of-1 indices on a DatetimeIndex of trading days.
  - Volatility and Sharpe are annualised with sqrt(252); 252 trading days/yr.
  - Risk-free rate is taken as zero (excess-return Sharpe == raw Sharpe). The
    deployed strategy and benchmarks are compared like-for-like, so a zero rf is
    a neutral choice and is labelled as such on the dashboard.
  - CAGR uses the ACTUAL calendar span (days/365.25), never an assumed year count.
  - Capture ratios use MONTHLY returns (Morningstar convention), benchmark sign
    defines up/down months.
  - All date anchoring (MTD/QTD/YTD/1Y) is done with pandas Timestamps, never by
    manual day arithmetic. pandas/Python months are 1-indexed.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config import TRADING_DAYS_PER_YEAR

AnnFactor = math.sqrt(TRADING_DAYS_PER_YEAR)


# --- construction ----------------------------------------------------------
def equity_series(dates: list[str], equity: list[float]) -> pd.Series:
    """Build a clean growth-of-1 Series from parallel date/equity lists."""
    s = pd.Series(equity, index=pd.to_datetime(dates), dtype="float64")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.dropna()


def daily_returns(eq: pd.Series) -> pd.Series:
    return eq.pct_change().dropna()


def drawdown_series(eq: pd.Series) -> pd.Series:
    """Drawdown from running peak, as a negative fraction."""
    return eq / eq.cummax() - 1.0


# --- summary statistics ----------------------------------------------------
def cagr(eq: pd.Series) -> float:
    if len(eq) < 2:
        return float("nan")
    days = (eq.index[-1] - eq.index[0]).days
    if days <= 0:
        return float("nan")
    years = days / 365.25
    return float(eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0


def ann_vol(eq: pd.Series) -> float:
    r = daily_returns(eq)
    return float(r.std(ddof=1) * AnnFactor) if len(r) > 1 else float("nan")


def sharpe(eq: pd.Series) -> float:
    r = daily_returns(eq)
    sd = r.std(ddof=1)
    if not sd or math.isnan(sd):
        return float("nan")
    return float(r.mean() / sd * AnnFactor)


def sortino(eq: pd.Series) -> float:
    r = daily_returns(eq)
    downside = r[r < 0]
    dd = downside.std(ddof=1)
    if not dd or math.isnan(dd):
        return float("nan")
    return float(r.mean() / dd * AnnFactor)


def max_drawdown(eq: pd.Series) -> float:
    return float(drawdown_series(eq).min()) if len(eq) else float("nan")


def calmar(eq: pd.Series) -> float:
    mdd = max_drawdown(eq)
    if not mdd or math.isnan(mdd):
        return float("nan")
    return cagr(eq) / abs(mdd)


def hit_rate(eq: pd.Series) -> float:
    r = daily_returns(eq)
    return float((r > 0).mean()) if len(r) else float("nan")


def summary_stats(eq: pd.Series) -> dict:
    """Full risk/return block for one equity series."""
    r = daily_returns(eq)
    return {
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) > 1 else float("nan"),
        "cagr": cagr(eq),
        "ann_vol": ann_vol(eq),
        "sharpe": sharpe(eq),
        "sortino": sortino(eq),
        "max_dd": max_drawdown(eq),
        "calmar": calmar(eq),
        "hit_rate": hit_rate(eq),
        "best_day": float(r.max()) if len(r) else float("nan"),
        "worst_day": float(r.min()) if len(r) else float("nan"),
        "current_dd": float(drawdown_series(eq).iloc[-1]) if len(eq) else float("nan"),
        "n_days": int(len(eq)),
        "start": eq.index[0].strftime("%Y-%m-%d"),
        "end": eq.index[-1].strftime("%Y-%m-%d"),
    }


# --- period anchors (MTD / QTD / YTD / 1Y / SI) ----------------------------
def _return_since(eq: pd.Series, anchor: pd.Timestamp) -> float | None:
    """Return from the last close on/before `anchor` to the latest close."""
    prior = eq[eq.index <= anchor]
    base = prior.iloc[-1] if len(prior) else eq.iloc[0]
    return float(eq.iloc[-1] / base - 1.0)


def period_returns(eq: pd.Series) -> dict:
    """MTD/QTD/YTD/1Y/SI returns, anchored with pandas Timestamps.

    Each anchor is the last trading day strictly before the period start, so the
    return measures the period itself (e.g. YTD anchors on prior-year-end close).
    """
    if len(eq) < 2:
        return {k: None for k in ("MTD", "QTD", "YTD", "1Y", "SI")}
    last = eq.index[-1]
    # Period-start boundaries (month is 1-indexed). Anchor = day before boundary.
    month_start = pd.Timestamp(year=last.year, month=last.month, day=1)
    q_first_month = 3 * ((last.month - 1) // 3) + 1
    quarter_start = pd.Timestamp(year=last.year, month=q_first_month, day=1)
    year_start = pd.Timestamp(year=last.year, month=1, day=1)
    one_year_ago = last - pd.DateOffset(years=1)
    return {
        "MTD": _return_since(eq, month_start - pd.Timedelta(days=1)),
        "QTD": _return_since(eq, quarter_start - pd.Timedelta(days=1)),
        "YTD": _return_since(eq, year_start - pd.Timedelta(days=1)),
        "1Y": _return_since(eq, one_year_ago),
        "SI": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
    }


# --- rolling series --------------------------------------------------------
def rolling_return(eq: pd.Series, window: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    return (eq / eq.shift(window) - 1.0).dropna()


def rolling_vol(eq: pd.Series, window: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    return (daily_returns(eq).rolling(window).std(ddof=1) * AnnFactor).dropna()


# --- monthly matrix --------------------------------------------------------
def monthly_returns(eq: pd.Series) -> pd.Series:
    """Calendar-month total returns (compounded from daily)."""
    return eq.resample("ME").last().pct_change().dropna()


def monthly_matrix(eq: pd.Series) -> dict:
    """{year: {1..12: ret, 'YEAR': annual ret}} for a heatmap."""
    m = monthly_returns(eq)
    out: dict[int, dict] = {}
    for ts, val in m.items():
        out.setdefault(ts.year, {})[ts.month] = float(val)  # month 1-indexed
    # Annual compounding per calendar year from the monthly series.
    for yr in list(out.keys()):
        ann = 1.0
        for mo in range(1, 13):
            if mo in out[yr]:
                ann *= 1.0 + out[yr][mo]
        out[yr]["YEAR"] = ann - 1.0
    return out


# --- benchmark-relative ----------------------------------------------------
def capture_ratios(model: pd.Series, bench: pd.Series) -> dict:
    """Up/down capture vs a benchmark, Morningstar monthly convention."""
    mm = monthly_returns(model)
    bm = monthly_returns(bench)
    join = pd.concat({"m": mm, "b": bm}, axis=1).dropna()
    if join.empty:
        return {"up_capture": None, "down_capture": None, "beta": None}

    def _cap(mask: pd.Series) -> float | None:
        sub = join[mask]
        if len(sub) < 2:
            return None
        m_c = float(np.prod(1.0 + sub["m"]) - 1.0)
        b_c = float(np.prod(1.0 + sub["b"]) - 1.0)
        return m_c / b_c if b_c else None

    up = _cap(join["b"] > 0)
    down = _cap(join["b"] < 0)
    # Beta from daily returns over the common window.
    dm = daily_returns(model)
    db = daily_returns(bench)
    dj = pd.concat({"m": dm, "b": db}, axis=1).dropna()
    var_b = dj["b"].var(ddof=1)
    beta = float(dj["m"].cov(dj["b"]) / var_b) if var_b else None
    return {"up_capture": up, "down_capture": down, "beta": beta}
