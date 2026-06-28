"""Navigo's own daily mark-to-market valuation layer (Phase 1, flag-gated).

Target architecture (DESIGN.md): engines *generate* (weekly target weights + the
weekly NAV anchor), Navigo *values* (the daily mark). This module extends the
engine's weekly NAV anchor forward by marking the published ``effective_weights``
— held FIXED from ``anchor_date`` (no intra-week trading) — to each subsequent
available close, in the portfolio's base currency (USD), converting non-USD
holdings via FX.

It is a *pure* transformation: the pipeline injects the close panel and FX series
(mirroring how benchmarks are injected into ``adapter``), so the module stays
unit-testable offline with no network. It NEVER generates a signal or a weight;
it only values the weights the engine published — the ``CLAUDE.md`` hard rule.

The three ways a self-marking layer is silently wrong, and the guard each carries
(DESIGN.md, "The discipline it must carry from day one"):

1. **Cost / FX replication drift.** The engine marks in USD, converting EUR
   (``.DE``) and CNY (``.SZ``) holdings to USD via FX, and applies NO cost within
   the week (the weights do not trade between anchors). This module replicates
   exactly that: FX-converted USD marks, zero within-week cost. The reconciliation
   gate (``tests/test_valuation.py``) proves it ties to the engine's own
   ``live_equity``; a within-week mismatch is a bug, not a cost difference. The
   weekly anchor handoff (where the 5 bps round-trip cost *does* bite) is a later
   phase.
2. **Marking stale weights.** ``weights_as_of`` (= ``anchor_date``, from the
   engine — may lag) is carried SEPARATELY from ``nav_as_of`` (= the last marked
   close, current) and never collapsed; the business-day gap between them is
   reported so the Phase-2 weights-freshness budget is a trivial addition.
3. **Look-ahead / silent splice.** The mark extends only STRICTLY forward from
   ``anchor_date``, only on dates with an available close, all date arithmetic via
   pandas / ``numpy.busday_count``. The result is a distinct block — it is never
   concatenated into the backtest curve here (the dashed-segment invariant).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

# FX is keyed by the holding's local currency; the series convert local -> base
# (USD). USD holdings need no conversion. The currency of a holding is read off
# the ticker/tradeAs suffix, the same convention adapter.build_holdings_prices
# already uses: a German listing (".DE") is priced in EUR, a Shenzhen listing
# (".SZ") in CNY, everything else in USD.
_SUFFIX_CCY = {".DE": "EUR", ".SZ": "CNY"}


def holding_currency(ticker: str, trade_as: str | None = None) -> str:
    trade_as = trade_as or ticker
    if str(trade_as).endswith(".DE"):
        return "EUR"
    if str(ticker).endswith(".SZ") or str(trade_as).endswith(".SZ"):
        return "CNY"
    return "USD"


def _close_for(closes: dict, ticker: str, trade_as: str | None):
    """The close series for a holding, by ticker then tradeAs (prices.py order)."""
    s = closes.get(ticker)
    if s is None and trade_as:
        s = closes.get(trade_as)
    return s


def mark_to_market(live: dict, closes: dict, fx: dict, registry: dict,
                   *, asof: str | None = None, base_currency: str | None = None) -> dict:
    """Mark the engine's fixed anchor weights forward to each available close.

    Parameters
    ----------
    live : the engine's live_track.json (uses anchor_date, anchor_equity,
        effective_weights — the headline NAV weights, EEM-tilt / breadth-gate
        aware, as the engine itself marks them).
    closes : {ticker_or_tradeAs -> pd.Series of local-currency close indexed by
        Timestamp}. The pipeline supplies a fresh panel; tests supply a fixture.
    fx : {currency -> pd.Series of local->USD rate indexed by Timestamp} for every
        non-base currency present (e.g. "EUR", "CNY"). USD needs no entry.
    asof : optional cap on the latest mark date (defaults to the latest available
        base-currency close). Marks are always strictly after anchor_date.

    Returns a dict carrying the two as-of stamps distinctly (never collapsed), the
    forward NAV extension, and a coverage report. NaN/missing coverage is flagged,
    never silently renormalised away — an uncovered weight understates the mark.
    """
    meta = registry.get("etf_meta", {})
    base_ccy = base_currency or registry.get("base_currency", "USD")
    anchor = pd.Timestamp(live["anchor_date"])
    anchor_equity = float(live["anchor_equity"])
    eff = live["effective_weights"]

    # Per-holding USD price series, aligned to one date index. The mark axis is the
    # UNION of every holding's trading calendar: the portfolio is re-marked on any
    # date a constituent can be repriced, with markets shut that day forward-filled
    # to their last close (exactly a mark-to-market). This matters across uneven
    # calendars — e.g. on US-only holidays (Juneteenth) the engine still marks,
    # because the European and Chinese holdings traded; a US-only axis would drop
    # that day and silently understate a real NAV move.
    all_dates: set = set()
    for tkr in eff:
        trade_as = meta.get(tkr, {}).get("tradeAs", tkr)
        s = _close_for(closes, tkr, trade_as)
        if s is not None and len(s):
            all_dates |= set(pd.DatetimeIndex(s.index))

    asof_ts = pd.Timestamp(asof) if asof else (max(all_dates) if all_dates else anchor)
    mark_dates = sorted(d for d in all_dates if d > anchor and d <= asof_ts)
    grid = pd.DatetimeIndex([anchor] + mark_dates)   # anchor first, for the return base

    covered_weight, uncovered = 0.0, []
    usd_ratio = {}                              # ticker -> series of (price*fx) / anchor(price*fx)
    for tkr, w in eff.items():
        trade_as = meta.get(tkr, {}).get("tradeAs", tkr)
        s = _close_for(closes, tkr, trade_as)
        if s is None or not len(s):
            uncovered.append(tkr)
            continue
        ccy = holding_currency(tkr, trade_as)
        usd = pd.Series(s, dtype="float64").reindex(grid, method="ffill")
        if ccy != base_ccy:
            rate = fx.get(ccy)
            if rate is None or not len(rate):
                uncovered.append(tkr)          # cannot convert -> cannot value honestly
                continue
            usd = usd * pd.Series(rate, dtype="float64").reindex(grid, method="ffill")
        base_val = usd.loc[anchor]
        if pd.isna(base_val) or base_val == 0:
            uncovered.append(tkr)              # no price on/before the anchor
            continue
        usd_ratio[tkr] = usd / base_val
        covered_weight += float(w)

    # NAV extension = anchor_equity * sum_i w_i * (usd_i[t] / usd_i[anchor]).
    equity = []
    for d in mark_dates:
        acc = 0.0
        for tkr, ratio in usd_ratio.items():
            r = ratio.loc[d]
            if not pd.isna(r):
                acc += float(eff[tkr]) * float(r)
        equity.append(round(anchor_equity * acc, 6))

    weights_as_of = live["anchor_date"]
    nav_as_of = mark_dates[-1].strftime("%Y-%m-%d") if mark_dates else weights_as_of
    weights_age_bdays = int(np.busday_count(
        dt.date.fromisoformat(weights_as_of), dt.date.fromisoformat(nav_as_of)))

    return {
        "base_currency": base_ccy,
        "anchor_date": weights_as_of,
        "anchor_equity": round(anchor_equity, 6),
        # The two as-of stamps, carried separately and NEVER collapsed (guard #2).
        "weights_as_of": weights_as_of,
        "nav_as_of": nav_as_of,
        "weights_age_bdays": weights_age_bdays,
        "dates": [d.strftime("%Y-%m-%d") for d in mark_dates],
        "equity": equity,
        "coverage": {
            "n_holdings": len(eff),
            "covered_weight": round(covered_weight, 6),
            "uncovered": uncovered,
            "complete": not uncovered and abs(covered_weight - 1.0) < 1e-4,
        },
        "method": ("fixed-weight forward mark from the weekly anchor; base "
                   f"{base_ccy}; non-base holdings FX-converted; no within-week cost"),
    }


def reconcile(mark: dict, live: dict) -> dict:
    """Compare Navigo's mark against the engine's own ``live_equity``.

    The whole point of Phase 1: prove the valuation logic reproduces the engine's
    daily mark. Returns absolute deviations in basis points on the overlapping
    dates. The freshest live date (= max(live_dates)) is the engine's run-day mark
    and the least-settled bar — its close can differ between the engine's fetch
    and ours by ordinary last-bar/timing noise — so deviations are reported split
    into ``settled`` (strictly before the freshest date) and the run-day bar, and
    the gate is taken on the settled segment.
    """
    eng = dict(zip(live["live_dates"], live["live_equity"]))
    nav = dict(zip(mark["dates"], mark["equity"]))
    overlap = [d for d in mark["dates"] if d in eng]
    freshest = max(live["live_dates"]) if live.get("live_dates") else None

    per_date = []
    for d in overlap:
        e, n = float(eng[d]), float(nav[d])
        dev_bps = (n / e - 1.0) * 1e4 if e else float("nan")
        per_date.append({"date": d, "engine": round(e, 6), "navigo": round(n, 6),
                         "dev_bps": round(dev_bps, 3), "settled": d < freshest})

    settled = [r["dev_bps"] for r in per_date if r["settled"]]
    alldev = [r["dev_bps"] for r in per_date]

    def _stats(xs):
        a = np.abs(np.array(xs, dtype="float64")) if xs else np.array([])
        return {"n": int(len(a)),
                "max_abs_bps": round(float(a.max()), 3) if len(a) else None,
                "mean_abs_bps": round(float(a.mean()), 3) if len(a) else None}

    return {
        "overlap_dates": overlap,
        "freshest_date": freshest,
        "per_date": per_date,
        "all": _stats(alldev),
        "settled": _stats(settled),
        "coverage_complete": bool(mark.get("coverage", {}).get("complete")),
    }


# --- production-path panel fetch (network; used only when the flag is on) ------
def fetch_panel(live: dict, registry: dict):
    """Fetch a fresh close + FX panel for the effective_weights holdings.

    yfinance lives here, isolated from the pure functions above. Returns
    ``(closes, fx)`` ready to pass to :func:`mark_to_market`, or ``(None, None)``
    if yfinance is unavailable. Used by the pipeline only when the valuation flag
    is enabled; the unit tests never touch the network (they inject a fixture).
    """
    try:
        import yfinance as yf
    except Exception as exc:                                   # pragma: no cover
        print(f"  [valuation] yfinance unavailable ({exc!r}); skipping mark")
        return None, None

    meta = registry.get("etf_meta", {})
    eff = live.get("effective_weights", {})
    tickers = sorted({meta.get(t, {}).get("tradeAs", t) for t in eff})
    ccys = {holding_currency(t, meta.get(t, {}).get("tradeAs", t)) for t in eff}
    fx_tickers = {"EUR": "EURUSD=X", "CNY": "CNY=X"}           # CNY=X is USDCNY -> invert
    want_fx = [fx_tickers[c] for c in ccys if c in fx_tickers]

    anchor = pd.Timestamp(live["anchor_date"])
    start = (anchor - pd.Timedelta(days=12)).strftime("%Y-%m-%d")
    try:
        raw = yf.download(tickers + want_fx, start=start, auto_adjust=True,
                          progress=False, threads=False)
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    except Exception as exc:                                   # pragma: no cover
        print(f"  [valuation] price fetch failed ({exc!r}); skipping mark")
        return None, None

    closes = {t: close[t].dropna() for t in tickers if t in close.columns}
    fx = {}
    if "EURUSD=X" in close.columns:
        fx["EUR"] = close["EURUSD=X"].dropna()
    if "CNY=X" in close.columns:
        fx["CNY"] = (1.0 / close["CNY=X"]).dropna()           # USDCNY -> CNYUSD
    return closes, fx


def build(live: dict, registry: dict) -> dict | None:
    """Pipeline entry point (flag-gated): fetch a fresh panel, mark, reconcile.

    Returns the mark dict with a nested ``reconcile`` block, or ``None`` if no
    panel could be fetched. This is attached to the dataset under a distinct
    ``valuation`` key — it never overwrites ``meta.asOf`` or the thin-renderer
    ``equity`` block, so a self-mark can never silently present as the headline.
    """
    closes, fx = fetch_panel(live, registry)
    if not closes:
        return None
    mark = mark_to_market(live, closes, fx, registry)
    if live.get("live_dates") and live.get("live_equity"):
        mark["reconcile"] = reconcile(mark, live)
    return mark
