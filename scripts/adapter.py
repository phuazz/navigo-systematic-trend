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

HORIZONS = [("1D", 1), ("1W", 5), ("1M", 21)]  # YTD / SI handled separately


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

    equity = {
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "model": [_r(v) for v in model_bt.values],
        "gate_only": _align(_series(gated), idx),
        "ungated": _align(_series(ungated), idx),
        "sleeves": sleeves,
        "benchmarks": benchmarks or {},
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
def _ret_over(s: pd.Series, bars: int) -> float | None:
    if len(s) <= bars:
        return None
    return float(s.iloc[-1] / s.iloc[-1 - bars] - 1.0)


def _ret_ytd(s: pd.Series) -> float | None:
    if len(s) < 2:
        return None
    yr_start = pd.Timestamp(year=s.index[-1].year, month=1, day=1)
    prior = s[s.index < yr_start]
    base = prior.iloc[-1] if len(prior) else s.iloc[0]
    return float(s.iloc[-1] / base - 1.0)


def build_attribution(equity, weights, prices, registry) -> dict:
    idx = pd.to_datetime(equity["dates"])
    sleeves = registry["sleeves"]
    tilt_on = weights["tilt_on"]

    # Sleeve-level: allocation-weighted sleeve return (approximate decomposition).
    sleeve_attr = {}
    for code, eq_list in equity["sleeves"].items():
        s = pd.Series(eq_list, index=idx, dtype="float64").dropna()
        cfg = sleeves[code]
        alloc = cfg.get("alloc_tilt" if tilt_on else "alloc", 0.0)
        row = {"name": cfg["name"], "alloc": _r(alloc, 4), "color": cfg.get("color")}
        for hk, bars in HORIZONS:
            rr = _ret_over(s, bars)
            row[hk] = {"ret": _r(rr, 5), "contrib": _r(alloc * rr, 5) if rr is not None else None}
        ry, rs = _ret_ytd(s), float(s.iloc[-1] / s.iloc[0] - 1.0)
        row["YTD"] = {"ret": _r(ry, 5), "contrib": _r(alloc * ry, 5) if ry is not None else None}
        row["SI"] = {"ret": _r(rs, 5), "contrib": _r(alloc * rs, 5)}
        sleeve_attr[code] = row

    # ETF-level: current weight x proxy return over window (current-weight approx).
    etf_attr = []
    for r in weights["rows"]:
        w = r["weight"] or 0.0
        s = prices.get(r["ticker"])
        if s is None:
            s = prices.get(r["tradeAs"])
        if s is None or w <= 1e-4:
            continue
        row = {"ticker": r["ticker"], "name": r["name"], "sleeve": r["sleeve"],
               "weight": _r(w, 5), "covered": True}
        for hk, bars in HORIZONS:
            rr = _ret_over(s, bars)
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
