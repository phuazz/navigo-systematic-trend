"""Recent price panel for the current holdings (attribution + trend signals).

The engine's holdings_prices_1y.json covers the broad-market, commodity and
thematic proxies (sleeves B and C) but NOT the US/Europe sector proxies
(sleeves A and D). Those are supplemented best-effort via yfinance on the
tradeAs ticker. A yfinance miss leaves a holding 'uncovered' (flagged on the
dashboard) rather than blocking the build.

Returns:
  series : {ticker -> pd.Series of close} for return-over-window attribution.
  meta   : {ticker -> {vs_ma200, change_pct}} for the Signals tab.
"""
from __future__ import annotations

import pandas as pd

try:
    import yfinance as yf
    _HAS_YF = True
except Exception:  # pragma: no cover
    _HAS_YF = False


def build_prices(holdings_prices_1y: dict, weights_rows: list, registry: dict,
                 *, fetch_missing: bool = True) -> tuple[dict, dict]:
    series: dict[str, pd.Series] = {}
    meta: dict[str, dict] = {}

    hp = (holdings_prices_1y or {}).get("prices", {})
    for tkr, d in hp.items():
        if d.get("dates") and d.get("prices"):
            s = pd.Series(d["prices"], index=pd.to_datetime(d["dates"]), dtype="float64").dropna()
            if len(s):
                series[tkr] = s
                meta[tkr] = {"vs_ma200": d.get("vs_ma200"), "change_pct": d.get("change_pct")}

    # Which active holdings still lack coverage?
    needed = []
    for r in weights_rows:
        if (r.get("weight") or 0) <= 1e-4:
            continue
        if r["ticker"] in series or r.get("tradeAs") in series:
            continue
        needed.append(r["tradeAs"])
    needed = sorted(set(needed))

    if fetch_missing and needed and _HAS_YF:
        try:
            raw = yf.download(needed, period="2y", auto_adjust=True, progress=False, threads=False)
            close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(
                columns={"Close": needed[0]})
            for tkr in needed:
                if tkr not in close:
                    continue
                s = close[tkr].dropna()
                if len(s) < 50:
                    continue
                series[tkr] = s
                ma200 = s.rolling(200).mean().iloc[-1] if len(s) >= 200 else s.mean()
                last, prev = s.iloc[-1], (s.iloc[-2] if len(s) > 1 else s.iloc[-1])
                meta[tkr] = {
                    "vs_ma200": float(last / ma200 - 1.0) if ma200 else None,
                    "change_pct": float(last / prev - 1.0) if prev else None,
                }
        except Exception as exc:
            print(f"  [prices] yfinance supplement failed ({exc!r}); some holdings uncovered")

    return series, meta
