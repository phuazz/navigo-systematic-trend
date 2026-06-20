"""Benchmark curves (SPY and 60/40 SPY/IEF) aligned to the model's date axis.

The engine does not publish a long benchmark history, so the monitor owns its
benchmark definition and fetches it via yfinance. Both benchmarks are rebased to
1.0 on the model's first date so they overlay the model equity directly.

Design choice: SPY is the honest all-equity hurdle ("did the tactical book beat
just holding stocks?"); 60/40 is the multi-asset hurdle. The 60/40 is a daily
constant-mix (rebalanced every day to 60/40), which slightly flatters vs a
drifting mix but is the standard reference and is labelled as such.

Robustness: a yfinance failure must NOT kill the build. We return ok=False and
the dashboard renders model-only with a flagged, missing benchmark feed.
"""
from __future__ import annotations

import pandas as pd

try:
    import yfinance as yf
    _HAS_YF = True
except Exception:  # pragma: no cover - yfinance always present in our env
    _HAS_YF = False


def _download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Adjusted-close frame indexed by date, one column per ticker."""
    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True,
        progress=False, threads=False,
    )
    if raw is None or len(raw) == 0:
        raise RuntimeError("yfinance returned no rows")
    # Single ticker -> flat columns; multi -> MultiIndex with 'Close' level.
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]] if "Close" in raw else raw
        close.columns = [tickers[0]]
    return close.dropna(how="all")


def build_benchmarks(model_dates: list[str], registry: dict) -> tuple[dict, bool, str]:
    """Return ({key: {dates, equity}}, ok, note) aligned to model_dates."""
    bms = registry.get("benchmarks", {})
    if not bms:
        return {}, True, "no benchmarks configured"
    if not _HAS_YF:
        return {}, False, "yfinance not installed"

    idx = pd.to_datetime(sorted(set(model_dates)))
    start = (idx[0] - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end = (idx[-1] + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    # Collect every raw ticker referenced by any benchmark.
    tickers: set[str] = set()
    for cfg in bms.values():
        if cfg["type"] == "yfinance":
            tickers.add(cfg["ticker"])
        elif cfg["type"] == "blend":
            tickers.update(cfg["components"].keys())

    try:
        close = _download(sorted(tickers), start, end)
    except Exception as exc:
        return {}, False, f"yfinance fetch failed: {exc}"

    # Reindex every raw series onto the model's trading days (ffill gaps).
    px = close.reindex(close.index.union(idx)).ffill().reindex(idx)
    if px.isna().all().any():
        missing = [c for c in px.columns if px[c].isna().all()]
        return {}, False, f"benchmark tickers had no data: {missing}"

    out: dict = {}
    for key, cfg in bms.items():
        if cfg["type"] == "yfinance":
            series = px[cfg["ticker"]].ffill()
            eq = series / series.iloc[0]
        else:  # daily constant-mix blend
            rets = px[list(cfg["components"])].pct_change().fillna(0.0)
            weights = pd.Series(cfg["components"])
            blended = (rets[weights.index] * weights).sum(axis=1)
            eq = (1.0 + blended).cumprod()
            eq = eq / eq.iloc[0]
        out[key] = {
            "dates": [d.strftime("%Y-%m-%d") for d in idx],
            "equity": [round(float(v), 6) for v in eq.values],
        }
    return out, True, "ok"
