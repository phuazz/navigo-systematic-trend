"""Benchmark curves aligned to the model's date axis.

Two sources, one basis rule (2026-09-06):

* ``type: "engine"`` — the S&P 500. The engine commits ``data/benchmark_spy.json``
  (adjusted closes from its sleeve-B price cache: the same series its weekly
  email and factsheet compare the deployed blend against). That file is the
  BASE of this dashboard's SPY curve, so both surfaces carry one S&P figure by
  construction. yfinance supplies only the EXTENSION: sessions after the
  export's last date, chained as daily returns onto the export's last value.
  The export refreshes with the engine's local scheduled runs (Tue/Wed/Sat/
  Sun), so mid-week the extension is one to three sessions long and the
  weekend build needs none.
* ``type: "yfinance"`` / ``"blend"`` — everything else, adjusted closes via
  yfinance as before. A blend leg on an engine-type ticker uses the
  engine-based column, so every S&P-derived figure on the page shares one
  basis.

Tail rule, from the 2026-09-05 build: yfinance withdraws the newest US bar for
a period overnight (the cycle the engine documents for European lines). That
build fetched at 01:22 UTC on the Saturday, received no 4 Sep SPY bar,
forward-filled Thursday's close onto Friday's date and printed S&P YTD +13.1%
against the model's Friday mark; the engine's email, on the same series with
Friday's bar, said +12.7%. A benchmark series therefore now ENDS at its own
last served close — never forward-filled onto model dates it has no bar for —
and carries ``asOf`` so the adapter and Data Health can say when it trails the
NAV. Interior gaps (a model date with no benchmark bar) are still forward-
filled and counted in ``interior_filled``.

Robustness: a yfinance failure must NOT kill the build. We return ok=False and
the dashboard renders model-only with a flagged, missing benchmark feed. An
engine-type benchmark whose export is absent from the bundle falls back to
yfinance for its whole history and says so in ``source``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    _HAS_YF = True
except Exception:  # pragma: no cover - yfinance always present in our env
    _HAS_YF = False

YF_BASIS = "yfinance adjusted close"


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


# --- engine export ---------------------------------------------------------
def engine_series(blob: dict | None) -> pd.Series | None:
    """The engine's committed benchmark export as a clean close series."""
    if not blob or not blob.get("dates") or not blob.get("closes"):
        return None
    s = pd.Series(blob["closes"], index=pd.to_datetime(blob["dates"]), dtype="float64")
    s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    return s if len(s) else None


def extend_with_returns(base: pd.Series, fresh: pd.Series | None) -> tuple[pd.Series, dict]:
    """Chain ``fresh``'s daily returns after ``base``'s last date onto ``base``'s last value.

    The base keeps its own basis and scale; the extension carries only the
    returns of sessions the base does not yet have. No extension when
    ``fresh`` lacks the base's last date (nothing to chain from) or has nothing
    newer — the series then simply ends where the export ends.
    """
    last = base.index[-1]
    info = {"from": last.strftime("%Y-%m-%d"), "sessions": 0, "note": None}
    if fresh is None or not len(fresh):
        info["note"] = "no extension series"
        return base, info
    fresh = fresh.dropna().sort_index()
    if last not in fresh.index:
        info["note"] = f"extension series has no bar on {info['from']}; not chained"
        return base, info
    tail = fresh[fresh.index > last]
    if not len(tail):
        return base, info
    chained = tail / float(fresh.loc[last]) * float(base.iloc[-1])
    info["sessions"] = int(len(tail))
    return pd.concat([base, chained]), info


# --- alignment -------------------------------------------------------------
def align_to_axis(series: pd.Series, idx: pd.DatetimeIndex) -> tuple[pd.Series, pd.Timestamp, int]:
    """Place ``series`` on the model axis: interior gaps forward-filled and
    counted, the tail beyond the series' own last bar left empty."""
    s = series.dropna().sort_index()
    last = s.index[-1]
    al = s.reindex(s.index.union(idx)).ffill().reindex(idx)
    al[al.index > last] = np.nan
    interior = int(((~idx.isin(s.index)) & (idx <= last) & (idx >= s.index[0])).sum())
    return al, last, interior


def _rebase(series: pd.Series) -> pd.Series:
    s = series.dropna()
    return series / s.iloc[0] if len(s) else series


def _emit(idx: pd.DatetimeIndex, eq: pd.Series, as_of: pd.Timestamp, **prov) -> dict:
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "equity": [None if pd.isna(v) else round(float(v), 6) for v in eq.values],
        "asOf": as_of.strftime("%Y-%m-%d"),
        **prov,
    }


# --- build -----------------------------------------------------------------
def build_benchmarks(model_dates: list[str], registry: dict,
                     live_dates: list[str] | None = None,
                     engine_files: dict | None = None) -> tuple[dict, bool, str]:
    """Return ({key: {dates, equity, asOf, basis, source, ...}}, ok, note).

    Every curve sits on the model's date axis (backtest + live dates) and is
    rebased to 1.0 on its first value. ``engine_files`` is the fetched engine
    bundle (filename -> parsed JSON); an engine-type benchmark reads its base
    series from there.
    """
    bms = registry.get("benchmarks", {})
    if not bms:
        return {}, True, "no benchmarks configured"
    if not _HAS_YF:
        return {}, False, "yfinance not installed"

    idx = pd.to_datetime(sorted(set(model_dates) | set(live_dates or [])))
    start = (idx[0] - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end = (idx[-1] + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    # Collect every raw ticker referenced by any benchmark (including FX pairs).
    tickers: set[str] = set()
    for cfg in bms.values():
        if cfg["type"] in ("yfinance", "engine"):
            tickers.add(cfg["ticker"])
            if cfg.get("fx"):
                tickers.add(cfg["fx"])
        elif cfg["type"] == "blend":
            tickers.update(cfg["components"].keys())

    try:
        close = _download(sorted(tickers), start, end)
    except Exception as exc:
        return {}, False, f"yfinance fetch failed: {exc}"

    # Raw series per ticker. An engine-type ticker is REPLACED by the engine
    # export (+ yfinance extension) so a blend leg on it shares the basis.
    raw: dict[str, pd.Series] = {t: close[t].dropna() for t in close.columns
                                 if not close[t].isna().all()}
    prov: dict[str, dict] = {}
    notes: list[str] = []
    for key, cfg in bms.items():
        if cfg["type"] != "engine":
            continue
        t = cfg["ticker"]
        blob = (engine_files or {}).get(cfg["file"])
        base = engine_series(blob if isinstance(blob, dict) else None)
        if base is None:
            prov[t] = {"basis": YF_BASIS,
                       "source": f"yfinance fallback — engine export {cfg['file']} unavailable",
                       "extension": None}
            notes.append(f"{key}: engine export {cfg['file']} unavailable, yfinance fallback")
            continue
        series, ext = extend_with_returns(base, raw.get(t))
        raw[t] = series
        src = f"engine export {cfg['file']} to {ext['from']}"
        if ext["sessions"]:
            src += f" + yfinance returns for {ext['sessions']} session{'s' if ext['sessions'] != 1 else ''}"
        prov[t] = {"basis": f"engine export: {blob.get('basis') or 'adjusted close'}",
                   "source": src, "extension": ext,
                   "export_written_at_utc": blob.get("written_at_utc")}
        notes.append(f"{key}: {src}" + (f" ({ext['note']})" if ext.get("note") else ""))

    def _prov(t: str) -> dict:
        return dict(prov.get(t) or {"basis": YF_BASIS, "source": "yfinance", "extension": None})

    out: dict = {}
    skipped: list[str] = []
    for key, cfg in bms.items():
        try:
            if cfg["type"] in ("yfinance", "engine"):
                t = cfg["ticker"]
                if t not in raw or (cfg.get("fx") and cfg["fx"] not in raw):
                    skipped.append(key)
                    continue
                al, last, interior = align_to_axis(raw[t], idx)
                p = _prov(t)
                if cfg.get("fx"):                     # convert a local-currency index to USD
                    fx, fx_last, _ = align_to_axis(raw[cfg["fx"]], idx)
                    al = al * fx
                    last = min(last, fx_last)
                    al[al.index > last] = np.nan
                    p["source"] += f" × {cfg['fx']} (yfinance)"
                out[key] = _emit(idx, _rebase(al), last, interior_filled=interior, **p)
            else:  # daily constant-mix blend
                comps = list(cfg["components"])
                if any(c not in raw for c in comps):
                    skipped.append(key)
                    continue
                cols, lasts = {}, []
                for c in comps:
                    al, last, _ = align_to_axis(raw[c], idx)
                    cols[c] = al
                    lasts.append(last)
                frame = pd.DataFrame(cols)
                as_of = min(lasts)
                rets = frame.pct_change().fillna(0.0)
                weights = pd.Series(cfg["components"])
                eq = (1.0 + (rets[weights.index] * weights).sum(axis=1)).cumprod()
                eq[eq.index > as_of] = np.nan
                legs = ", ".join(f"{c}: {_prov(c)['source']}" for c in comps)
                out[key] = _emit(idx, _rebase(eq), as_of, interior_filled=0,
                                 basis="daily constant-mix of adjusted closes",
                                 source=legs, extension=None)
        except Exception:
            skipped.append(key)

    if not out:
        return {}, False, f"no benchmark had usable data (skipped {skipped})"
    note = "ok" if not skipped else f"ok; skipped {skipped}"
    if notes:
        note += "; " + "; ".join(notes)
    return out, True, note
