"""Normalise the engine's raw feeds into the monitor's data contract.

Pure transformation: takes the fetched bundle + registry (+ optional benchmarks
and previous dataset) and returns one JSON-serialisable dict. No network here —
the pipeline injects benchmarks so the adapter stays unit-testable offline.

The contract is documented in README.md; the headline equity is the deployed,
gate-and-tilt overlay applied curve, never silently spliced with the live
mark-to-market extension.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import metrics
from config import TRADING_DAYS_PER_YEAR as TRADING_DAYS

HZ_KEYS = ("1D", "1W", "1M")  # YTD / SI handled separately


def _anchors(ref) -> dict:
    """Calendar-anchored window starts for a reference date.

    1-Day anchors on the previous calendar weekday (skipping weekends) so a
    single session is measured consistently across markets and a non-trading
    reference day reads flat — rather than each series' immediately-preceding
    bar, which would span holidays unevenly across a multi-market book.
    """
    ref = pd.Timestamp(ref)
    return {"1D": metrics.prev_weekday(ref),
            "1W": ref - pd.Timedelta(days=7),
            "1M": ref - pd.DateOffset(months=1)}


# --- small helpers ---------------------------------------------------------
def _r(x, n=6):
    """Round floats for compact JSON; pass through None/non-finite as None."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(f) else round(f, n)


def _series(d: dict) -> pd.Series:
    return metrics.equity_series(d["dates"], d["equity"])


def _align(s: pd.Series, idx: pd.DatetimeIndex) -> list:
    return [_r(v) for v in s.reindex(idx).ffill().values]


# --- equity ----------------------------------------------------------------
def build_equity(live, multi, overlay, registry, benchmarks) -> tuple[dict, pd.Series, pd.Series]:
    src = registry["source"]
    deployed = overlay["gated_variants"][src["deployed_key"]]
    gated = overlay["gated_variants"][src["gated_key"]]
    ungated = multi["strategies"][src["ungated_key"]]

    model_bt = _series(deployed)               # deployed backtest (overlay applied)
    idx = model_bt.index

    sleeves = {}
    for code, cfg in registry["sleeves"].items():
        key = cfg.get("key")
        if key and key in multi["strategies"]:
            sleeves[code] = _align(_series(multi["strategies"][key]), idx)

    # Live mark-to-market extension — kept as a SEPARATE, flagged segment.
    live_block = None
    model_full = model_bt
    if live.get("live_dates") and live.get("live_equity"):
        anchor_d = pd.Timestamp(live["anchor_date"])
        anchor_e = float(live.get("anchor_equity", model_bt.iloc[-1]))
        lv = metrics.equity_series(live["live_dates"], live["live_equity"])
        lv = lv[lv.index > anchor_d]
        if len(lv):
            live_block = {
                "anchor_date": live["anchor_date"],
                "anchor_equity": _r(anchor_e),
                "dates": [d.strftime("%Y-%m-%d") for d in lv.index],
                "equity": [_r(v) for v in lv.values],
            }
            model_full = pd.concat([model_bt, lv])

    # Attach label + colour from the registry so the client benchmark selector
    # can render any benchmark consistently.
    bmeta = registry.get("benchmarks", {})
    bms = {k: {**v, "label": bmeta.get(k, {}).get("label", k),
               "color": bmeta.get(k, {}).get("color", "#8a8a82"),
               "default": bool(bmeta.get(k, {}).get("default"))}
           for k, v in (benchmarks or {}).items()}

    equity = {
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "model": [_r(v) for v in model_bt.values],
        "gate_only": _align(_series(gated), idx),
        "ungated": _align(_series(ungated), idx),
        "sleeves": sleeves,
        "benchmarks": bms,
        "live": live_block,
        "labels": {
            "model": deployed.get("label", "Deployed model"),
            "gate_only": gated.get("label", "Gate only"),
            "ungated": ungated.get("label", "Ungated reference"),
        },
    }
    return equity, model_bt, model_full


# --- weights ---------------------------------------------------------------
def build_weights(live, registry) -> dict:
    meta = registry["etf_meta"]
    sleeves = registry["sleeves"]
    tilt_on = bool(live.get("eem_tilt_active"))
    eff = live["effective_weights"]
    sleeve_ext = live.get("sleeve_extensions", {})

    def sleeve_alloc(code: str) -> float:
        c = sleeves.get(code, {})
        return float(c.get("alloc_tilt" if tilt_on else "alloc", 0.0))

    rows = []
    for tkr, w in eff.items():
        m = meta.get(tkr, {"name": tkr, "sleeve": "?", "assetClass": "?", "geo": "?", "theme": "?", "tradeAs": tkr})
        code = m["sleeve"]
        within = None
        build = []
        # sleeve_extensions is keyed by the engine sleeve key (e.g. 'strategy_b'),
        # not the short code ('B') — map through the registry.
        sk = sleeves.get(code, {}).get("key")
        if sk and sk in sleeve_ext and tkr in sleeve_ext[sk].get("weights", {}):
            within = float(sleeve_ext[sk]["weights"][tkr])
            alloc = sleeve_alloc(code)
            build.append({"sleeve": code, "alloc": _r(alloc, 4), "within": _r(within, 4),
                          "contrib": _r(alloc * within, 5)})
        if tkr == "EEM" and tilt_on:
            build.append({"sleeve": "TILT", "alloc": _r(sleeve_alloc("TILT"), 4),
                          "within": 1.0, "contrib": _r(sleeve_alloc("TILT"), 5)})
        if code == "CASH":
            build.append({"sleeve": "Overlay", "alloc": None, "within": None, "contrib": _r(w, 5)})
        rows.append({
            "ticker": tkr, "weight": _r(w, 5), "sleeve": code,
            "name": m["name"], "assetClass": m["assetClass"], "geo": m["geo"],
            "theme": m["theme"], "tradeAs": m["tradeAs"],
            "within_sleeve": _r(within, 4), "build": build,
        })
    rows.sort(key=lambda x: (x["weight"] or 0), reverse=True)

    def group(field: str) -> dict:
        out: dict[str, float] = {}
        for r in rows:
            out[r[field]] = out.get(r[field], 0.0) + (r["weight"] or 0.0)
        return {k: _r(v, 5) for k, v in sorted(out.items(), key=lambda kv: -kv[1])}

    weights_only = np.array([r["weight"] or 0.0 for r in rows])
    cash = sum(r["weight"] or 0.0 for r in rows if r["sleeve"] == "CASH")
    return {
        "tilt_on": tilt_on,
        "rows": rows,
        "by_sleeve": group("sleeve"),
        "by_asset_class": group("assetClass"),
        "by_geo": group("geo"),
        "by_theme": group("theme"),
        "concentration": {
            "hhi": _r(float((weights_only ** 2).sum()), 4),
            "top5": _r(float(np.sort(weights_only)[::-1][:5].sum()), 4),
            "n_holdings": int((weights_only > 1e-4).sum()),
        },
        "cash_pct": _r(cash, 5),
        "invested_pct": _r(1.0 - cash, 5),
        "sleeve_meta": {c: {"name": v["name"], "alloc": _r(sleeve_alloc(c), 4),
                            "color": v.get("color"), "desc": v.get("desc")}
                        for c, v in sleeves.items()},
    }


# --- regime ----------------------------------------------------------------
def build_regime(overlay, live) -> dict:
    gp = overlay["gate_parameters"]
    et = overlay.get("phase22_eem_tilt", {})
    ds = et.get("daily_series", {})
    eem_series = None
    if ds.get("dates"):
        eem_series = {
            "dates": ds["dates"],
            "ratio": [_r(v, 5) for v in ds.get("ratio", [])],
            "fast_ma": [_r(v, 5) for v in ds.get("fast_ma", [])],
            "slow_ma": [_r(v, 5) for v in ds.get("slow_ma", [])],
        }
    return {
        "state": overlay["current_state"],
        "since": overlay["current_state_since"],
        "breadth": _r(overlay["current_breadth"], 4),
        "panel_end_date": overlay.get("panel_end_date"),
        "off_threshold": _r(gp["off_threshold"], 3),
        "on_threshold": _r(gp["on_threshold"], 3),
        "derisk_fraction": _r(gp["derisk_fraction"], 3),
        "fallback_ticker": gp.get("fallback_ticker"),
        "switch_cost_bps": gp.get("switch_cost_bps"),
        "n_switches": overlay.get("n_switches"),
        "days_risk_off": overlay.get("days_risk_off"),
        "pct_days_risk_off": _r(overlay.get("pct_days_risk_off"), 2),
        "events": [{"date": e["date"], "direction": e["direction"], "breadth": _r(e.get("breadth"), 4)}
                   for e in overlay.get("events", [])],
        "live_state": live.get("regime_state"),
        "eem_tilt": {
            "enabled": et.get("enabled"),
            "state": et.get("current_state"),
            "since": et.get("current_state_since"),
            "ratio": _r(et.get("current_ratio"), 5),
            "fast_ma": _r(et.get("current_fast_ma"), 5),
            "slow_ma": _r(et.get("current_slow_ma"), 5),
            "n_switches": et.get("n_switches"),
            "pct_days_on": _r(et.get("pct_days_tilt_on"), 2),
            "active": bool(live.get("eem_tilt_active")),
            "series": eem_series,
        },
    }


# --- statistics ------------------------------------------------------------
def build_stats(model_bt, model_full, overlay, registry, benchmarks) -> dict:
    src = registry["source"]
    eng = overlay["gated_variants"][src["deployed_key"]]
    engine_stats = {k: eng.get(k) for k in ("sharpe", "cagr", "total_return", "max_dd")}

    # Value-of-the-overlay comparison, straight from the engine's own figures.
    keys = ("sharpe", "cagr", "total_return", "max_dd")
    overlay_compare = {
        "ungated": {k: _r(overlay["ungated_reference"].get(k)) for k in keys},
        "gate_only": {k: _r(overlay["gated_variants"][src["gated_key"]].get(k)) for k in keys},
        "deployed": {k: _r(eng.get(k)) for k in keys},
    }

    full = metrics.summary_stats(model_full)
    full["period_returns"] = metrics.period_returns(model_full)

    # Capture + period/summary stats vs each benchmark (on the backtest window).
    caps = {}
    bench_stats = {}
    for key, bm in (benchmarks or {}).items():
        bseries = metrics.equity_series(bm["dates"], bm["equity"])
        caps[key] = metrics.capture_ratios(model_bt, bseries)
        bench_stats[key] = {
            "period_returns": {k: _r(v, 5) for k, v in metrics.period_returns(bseries).items()},
            "cagr": _r(metrics.cagr(bseries)), "ann_vol": _r(metrics.ann_vol(bseries)),
            "sharpe": _r(metrics.sharpe(bseries)), "max_dd": _r(metrics.max_drawdown(bseries)),
        }

    # Reconcile our recompute (backtest window) against the engine's own figures.
    recomputed = metrics.summary_stats(model_bt)
    diffs = {
        "sharpe": _r(abs((recomputed["sharpe"] or 0) - (engine_stats["sharpe"] or 0)), 4),
        "cagr": _r(abs((recomputed["cagr"] or 0) - (engine_stats["cagr"] or 0)), 4),
        "total_return": _r(abs((recomputed["total_return"] or 0) - (engine_stats["total_return"] or 0)), 4),
        "max_dd": _r(abs((recomputed["max_dd"] or 0) - (engine_stats["max_dd"] or 0)), 4),
    }
    tol = {"sharpe": 0.05, "cagr": 0.01, "total_return": 0.05, "max_dd": 0.01}
    reconcile_ok = all((diffs[k] or 0) <= tol[k] for k in tol)

    return {
        **{k: _r(v) for k, v in full.items() if not isinstance(v, (dict, str))},
        "start": full["start"], "end": full["end"],
        "period_returns": {k: _r(v, 5) for k, v in full["period_returns"].items()},
        "capture": caps,
        "benchmark_stats": bench_stats,
        "engine_stats": {k: _r(v) for k, v in engine_stats.items()},
        "overlay_compare": overlay_compare,
        "reconcile": {"diffs": diffs, "tol": tol, "ok": bool(reconcile_ok),
                      "recomputed": {k: _r(recomputed[k]) for k in tol}},
    }


# --- attribution -----------------------------------------------------------
def _ret_ytd(s: pd.Series) -> float | None:
    if len(s) < 2:
        return None
    yr_start = pd.Timestamp(year=s.index[-1].year, month=1, day=1)
    prior = s[s.index < yr_start]
    base = prior.iloc[-1] if len(prior) else s.iloc[0]
    return float(s.iloc[-1] / base - 1.0)


def build_attribution(equity, weights, prices, registry, risk_by_ticker=None, ref_date=None) -> dict:
    idx = pd.to_datetime(equity["dates"])
    sleeves = registry["sleeves"]
    tilt_on = weights["tilt_on"]
    sref = idx[-1]                                   # sleeves: their own (backtest) last date
    sanc = _anchors(sref)
    eref = pd.Timestamp(ref_date) if ref_date else sref   # ETFs/benchmarks: global (live) date
    eanc = _anchors(eref)

    # Sleeve-level: allocation-weighted sleeve return (approximate decomposition).
    sleeve_attr = {}
    for code, eq_list in equity["sleeves"].items():
        s = pd.Series(eq_list, index=idx, dtype="float64").dropna()
        cfg = sleeves[code]
        alloc = cfg.get("alloc_tilt" if tilt_on else "alloc", 0.0)
        row = {"name": cfg["name"], "alloc": _r(alloc, 4), "color": cfg.get("color")}
        for hk in HZ_KEYS:
            rr = metrics.windowed_return(s, sref, sanc[hk])
            row[hk] = {"ret": _r(rr, 5), "contrib": _r(alloc * rr, 5) if rr is not None else None}
        ry, rs = _ret_ytd(s), float(s.iloc[-1] / s.iloc[0] - 1.0)
        row["YTD"] = {"ret": _r(ry, 5), "contrib": _r(alloc * ry, 5) if ry is not None else None}
        row["SI"] = {"ret": _r(rs, 5), "contrib": _r(alloc * rs, 5)}
        sleeve_attr[code] = row

    # ETF-level: current weight x proxy return over a calendar-anchored window.
    etf_attr = []
    for r in weights["rows"]:
        w = r["weight"] or 0.0
        s = prices.get(r["ticker"])
        if s is None:
            s = prices.get(r["tradeAs"])
        if s is None or w <= 1e-4:
            continue
        row = {"ticker": r["ticker"], "name": r["name"], "sleeve": r["sleeve"],
               "theme": r["theme"], "weight": _r(w, 5), "covered": True}
        rk = (risk_by_ticker or {}).get(r["ticker"])
        if rk:
            row["vol"], row["risk_pct"], row["ret_1y"] = rk["vol"], rk["risk_pct"], rk["ret_1y"]
        for hk in HZ_KEYS:
            rr = metrics.windowed_return(s, eref, eanc[hk])
            row[hk] = {"ret": _r(rr, 5), "contrib": _r(w * rr, 5) if rr is not None else None}
        ry = _ret_ytd(s)
        row["YTD"] = {"ret": _r(ry, 5), "contrib": _r(w * ry, 5) if ry is not None else None}
        etf_attr.append(row)
    # Holdings with no price coverage (sector/Europe proxies) — flag, do not hide.
    uncovered = [r["ticker"] for r in weights["rows"]
                 if (r["weight"] or 0) > 1e-4
                 and r["ticker"] not in prices and r["tradeAs"] not in prices]
    return {"sleeve": sleeve_attr, "etf": etf_attr, "uncovered": uncovered,
            "note": "Sleeve contribution = sleeve allocation x sleeve return (approximate; "
                    "residual is overlay/tilt/rebalance). ETF contribution uses current weight."}


# --- signals ---------------------------------------------------------------
def build_signals(weights, prices_meta, registry) -> dict:
    """Per-sleeve 'why these weights': current holdings + trend context where available."""
    out: dict[str, list] = {}
    for r in weights["rows"]:
        code = r["sleeve"]
        if code == "CASH" and (r["weight"] or 0) < 1e-4:
            continue
        ctx = prices_meta.get(r["ticker"]) or prices_meta.get(r["tradeAs"]) or {}
        out.setdefault(code, []).append({
            "ticker": r["ticker"], "name": r["name"], "theme": r["theme"],
            "weight": r["weight"], "within_sleeve": r["within_sleeve"],
            "vs_ma200": _r(ctx.get("vs_ma200"), 4) if ctx else None,
            "change_pct": _r(ctx.get("change_pct"), 4) if ctx else None,
        })
    for code in out:
        out[code].sort(key=lambda x: (x["weight"] or 0), reverse=True)
    return out


# --- changes (vs previous bake) --------------------------------------------
def build_changes(weights, prev_dataset) -> dict:
    cur = {r["ticker"]: (r["weight"] or 0.0) for r in weights["rows"]}
    if not prev_dataset or "weights" not in prev_dataset:
        return {"available": False, "turnover": None, "deltas": [], "asOf_prev": None}
    prev = {r["ticker"]: (r["weight"] or 0.0) for r in prev_dataset["weights"]["rows"]}
    tickers = set(cur) | set(prev)
    deltas = []
    for t in tickers:
        d = cur.get(t, 0.0) - prev.get(t, 0.0)
        if abs(d) > 5e-4:
            deltas.append({"ticker": t, "from": _r(prev.get(t, 0.0), 5),
                           "to": _r(cur.get(t, 0.0), 5), "delta": _r(d, 5)})
    deltas.sort(key=lambda x: abs(x["delta"]), reverse=True)
    turnover = 0.5 * sum(abs(cur.get(t, 0.0) - prev.get(t, 0.0)) for t in tickers)
    return {"available": True, "turnover": _r(turnover, 5), "deltas": deltas,
            "asOf_prev": (prev_dataset.get("meta") or {}).get("asOf")}


# --- rebalance / trade ledger (append-only) --------------------------------
def build_trades(ledger, weights, registry, asof):
    """Append-only rebalance ledger.

    Diffs the current target weights against the last recorded set; when turnover
    crosses a threshold a rebalance entry is recorded with its trades (entries,
    exits, adds, trims) and their weight deltas. History begins when monitoring
    started — the engine does not expose pre-monitoring target weights, so the
    first build seeds the opening position. Returns (new_ledger, trades_block).
    """
    meta = registry["etf_meta"]
    cur = {r["ticker"]: float(r["weight"] or 0.0) for r in weights["rows"] if (r["weight"] or 0) > 1e-4}
    ledger = ledger or {}
    last = ledger.get("last_weights")
    log = list(ledger.get("log", []))

    def nm(t):
        return meta.get(t, {}).get("name", t)

    def act(f, c):
        if f <= 1e-4 and c > 1e-4:
            return "NEW"
        if f > 1e-4 and c <= 1e-4:
            return "EXIT"
        return "ADD" if c > f else "TRIM"

    if last is None:                                   # first ever build — seed opening book
        deltas = [{"ticker": t, "name": nm(t), "from": 0.0, "to": _r(w, 5), "delta": _r(w, 5), "action": "INITIAL"}
                  for t, w in sorted(cur.items(), key=lambda x: -x[1])]
        log.append({"date": asof, "type": "initial", "turnover": _r(sum(cur.values()), 5),
                    "n": len(deltas), "deltas": deltas})
        last = cur
    else:
        tickers = set(cur) | set(last)
        raw = [(t, float(last.get(t, 0.0)), cur.get(t, 0.0)) for t in tickers]
        turnover = 0.5 * sum(abs(c - f) for _, f, c in raw)
        deltas = sorted(
            [{"ticker": t, "name": nm(t), "from": _r(f, 5), "to": _r(c, 5), "delta": _r(c - f, 5), "action": act(f, c)}
             for t, f, c in raw if abs(c - f) > 5e-4],
            key=lambda d: -abs(d["delta"] or 0))
        if turnover > 0.005 and deltas:                # ignore sub-threshold drift
            log.append({"date": asof, "type": "rebalance", "turnover": _r(turnover, 5),
                        "n": len(deltas), "deltas": deltas})
            last = cur

    new_ledger = {"last_weights": last, "updated": asof, "log": log}
    trades = {"since": log[0]["date"] if log else asof, "count": len(log),
              "log": list(reversed(log)), "asOf": asof}      # newest first for display
    return new_ledger, trades


# --- short-horizon P&L (model vs benchmarks) -------------------------------
def build_pnl(model_full: pd.Series, benchmarks: dict) -> dict:
    """1-day / 1-week / 1-month return for the model and each benchmark.

    All windows are calendar-anchored off one global reference date (the model's
    latest mark): 1-Day anchors on the previous calendar weekday, so it is a true
    single session and the benchmarks (which carry the live date) compare like for
    like. The model marks once daily at close, so there is no separate intraday
    figure — the 1-Day already reflects the latest mark-to-market session.
    """
    ref = model_full.index[-1]
    anc = _anchors(ref)

    def rets(s: pd.Series) -> dict:
        return {k: _r(metrics.windowed_return(s, ref, anc[k]), 5) for k in HZ_KEYS}

    out = {"model": rets(model_full)}
    for k, bm in (benchmarks or {}).items():
        out[k] = rets(metrics.equity_series(bm["dates"], bm["equity"]))
    sources = list(out.keys())
    return {p: {src: out[src][p] for src in sources} for p in HZ_KEYS}


# --- risk decomposition ----------------------------------------------------
def build_risk(price_series: dict, weights: dict, registry: dict) -> dict:
    """Per-holding annualised vol and risk contribution from the ~1y price panel.

    Risk contribution_i = w_i * (Sigma w)_i / sigma_p, summing to 100% of portfolio
    risk — the standard 'where does my risk actually sit vs my capital' view. The
    covariance uses the most recent ~252 aligned observations. European holdings
    are priced in local currency (a known, flagged approximation).
    """
    rows = [r for r in weights["rows"] if (r.get("weight") or 0) > 1e-4]
    series, info = {}, {}
    for r in rows:
        s = price_series.get(r["ticker"])
        if s is None:
            s = price_series.get(r["tradeAs"])
        if s is None:
            continue
        series[r["ticker"]] = s
        info[r["ticker"]] = r
    if len(series) < 2:
        return {"holdings": [], "by_ticker": {}, "port_vol": None,
                "note": "insufficient price coverage for a risk decomposition"}

    px = pd.concat(series, axis=1).dropna().tail(TRADING_DAYS)
    rets = px.pct_change().dropna()
    tickers = list(px.columns)
    cov = rets.cov().values * TRADING_DAYS                      # annualised covariance
    wv = np.array([info[t]["weight"] or 0.0 for t in tickers], dtype="float64")
    wv = wv / wv.sum() if wv.sum() else wv
    port_var = float(wv @ cov @ wv)
    port_vol = math.sqrt(port_var) if port_var > 0 else 0.0
    rc = wv * (cov @ wv)                                        # risk contribution (to variance)
    risk_pct = rc / port_var if port_var > 0 else np.zeros_like(rc)
    vols = np.sqrt(np.diag(cov))
    holdings, by_ticker = [], {}
    for i, t in enumerate(tickers):
        r = info[t]
        ret1y = float(px[t].iloc[-1] / px[t].iloc[0] - 1.0)
        rec = {"vol": _r(vols[i], 4), "risk_pct": _r(float(risk_pct[i]), 5), "ret_1y": _r(ret1y, 5)}
        by_ticker[t] = rec
        holdings.append({"ticker": t, "name": r["name"], "sleeve": r["sleeve"],
                         "theme": r["theme"], "weight": r["weight"], **rec})
    return {"holdings": holdings, "by_ticker": by_ticker, "port_vol": _r(port_vol, 4),
            "obs": int(len(rets)),
            "note": "Annualised from the most recent ~1y of daily returns. Risk contribution "
                    "= w*(Sigma w)/sigma_p, summing to 100%. European holdings are priced in "
                    "local currency."}


# --- per-holding price panel (expandable row charts) -----------------------
def build_holdings_prices(holdings_raw: dict, price_series: dict, weights: dict, *, tail=252) -> dict:
    """Close + 50/200-day MA series and trend signals per current holding.

    Engine-covered tickers reuse the engine's own MA series (full 200-day cover);
    the US/Europe sector proxies use the yfinance close panel with MAs computed
    here. Feeds the expandable price chart on the Allocation table.
    """
    hp = (holdings_raw or {}).get("prices", {})

    def _last(a):
        for v in reversed(a):
            if v is not None and (not isinstance(v, float) or math.isfinite(v)):
                return v
        return None

    out = {}
    for r in weights["rows"]:
        if (r.get("weight") or 0) <= 1e-4:
            continue
        t, tradeAs = r["ticker"], r.get("tradeAs", r["ticker"])
        src = hp.get(t) or hp.get(tradeAs)
        if src and src.get("ma200"):
            dates = src["dates"][-tail:]; close = src["prices"][-tail:]
            ma50 = (src.get("ma50") or [None] * len(src["prices"]))[-tail:]
            ma200 = (src.get("ma200") or [None] * len(src["prices"]))[-tail:]
        else:
            s = price_series.get(t)
            if s is None:
                s = price_series.get(tradeAs)
            if s is None:
                continue
            ma50s, ma200s = s.rolling(50).mean(), s.rolling(200).mean()
            dates = [d.strftime("%Y-%m-%d") for d in s.index[-tail:]]
            close = [float(v) for v in s.values[-tail:]]
            ma50 = [None if pd.isna(v) else float(v) for v in ma50s.values[-tail:]]
            ma200 = [None if pd.isna(v) else float(v) for v in ma200s.values[-tail:]]
        c, m50, m200 = _last(close), _last(ma50), _last(ma200)
        c0 = next((v for v in close if v is not None), None)
        out[t] = {
            "dates": dates,
            "close": [_r(v, 4) for v in close],
            "ma50": [_r(v, 4) for v in ma50],
            "ma200": [_r(v, 4) for v in ma200],
            "ccy": "EUR" if str(tradeAs).endswith(".DE") else ("CNY" if str(t).endswith(".SZ") else "USD"),
            "signals": {
                "above_200": bool(c is not None and m200 is not None and c > m200),
                "golden": bool(m50 is not None and m200 is not None and m50 > m200),
                "vs_ma200": _r(c / m200 - 1, 4) if (c and m200) else None,
                "mom_1y": _r(c / c0 - 1, 4) if (c and c0) else None,
            },
        }
    return out
