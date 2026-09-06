"""Fail-loud (and fail-visible) data-integrity gates.

This is the direct lesson of the engine's de-risk staleness incident: a panel can
go stale while the headline keeps printing a confident state. So the monitor
checks freshness and internal consistency on every build and surfaces the result
as a first-class object the dashboard renders (a STALE banner + a red Data Health
tab), rather than quietly trusting whatever it fetched.

Levels: 'ok' < 'warn' < 'stale'. Nothing here hard-stops the build (a partial,
clearly-flagged dashboard beats no dashboard); structural impossibilities have
already raised in the adapter. Business-day lags use numpy.busday_count so the
weekend never inflates a lag — no hand-rolled day arithmetic.

Two different freshness bases, deliberately (2026-08-24):

  - The weekly feeds (breadth/regime panel, strategy equity) keep plain
    business-day budgets. They trail by design, the slack is wide, and a
    holiday never matters at that width.
  - The live Price / NAV feed is judged against the TRUE last completed NYSE
    session instead. A business-day budget cannot express "one session behind"
    for a daily feed: on 2026-08-21 the engine's own guard blocked publication
    and the monitor baked Thursday's mark on a Friday — busday_count read 1 —
    while a perfectly healthy run whose 23:40 UTC cron slips past midnight also
    reads 1. No threshold separates those two, so any budget either misses the
    real miss or false-alarms nightly. The calendar anchor separates them
    cleanly, and it is the same measure check_capture_integrity.py and
    sentinel_check.py already use, so the page, the build guard and the
    outside-in sentinel now agree by construction rather than by coincidence.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from nyse_sessions import last_completed_session, sessions_behind

_RANK = {"ok": 0, "warn": 1, "stale": 2}


def _bday_lag(asof: str | None, run_date: dt.date) -> int | None:
    if not asof:
        return None
    a = dt.date.fromisoformat(asof[:10])
    return int(np.busday_count(a, run_date))


def run(bundle: dict, registry: dict, run_date: dt.date, stats: dict,
        bench_ok: bool, bench_note: str,
        now_utc: dt.datetime | None = None,
        benchmarks: dict | None = None) -> dict:
    """``now_utc`` anchors the live feed's session check. It is separate from
    ``run_date`` because the verdict needs the INSTANT, not the date: whether
    Friday's session has closed depends on the time of day, and run_date has
    already thrown that away. Defaults to the real clock; tests pass it.

    ``benchmarks`` is the built benchmark dict (each carrying ``asOf``); an
    engine-type benchmark gets a feed row judged against the NAV's own as-of."""
    fb = registry["freshness"]
    now_utc = now_utc or dt.datetime.now(dt.timezone.utc)
    live = bundle["live_track.json"]
    overlay = bundle["risk_overlay.json"]
    multi = bundle["multi_strategy.json"]

    price_asof = (live.get("live_dates") or [None])[-1] or live.get("anchor_date")
    regime_asof = overlay.get("panel_end_date")

    feeds = []

    def add(name, asof, lag, budget, computed, basis="business days", warn_at=None, **extra):
        """``warn_at``: a lag strictly above this is 'warn'. Defaults to
        ``budget - 2`` — the approach band that suits the wide weekly budgets.
        Pass ``warn_at=budget`` for a tight daily feed, which steps straight
        from ok to stale: at a budget of 0 there is no middle ground to occupy,
        and a permanent 'warn' on a healthy feed would train the eye to ignore
        the banner. ``extra`` keys ride along on the feed row."""
        level = "ok"
        if lag is None:
            level = "warn"
        elif lag > budget:
            level = "stale"
        elif lag > (budget - 2 if warn_at is None else warn_at):
            level = "warn"
        feeds.append({"feed": name, "asOf": asof, "bday_lag": lag, "budget_bdays": budget,
                      "level": level, "computed_at": computed, "basis": basis, **extra})

    # Live NAV: sessions behind the last completed NYSE session (see module
    # docstring). Budget 0 = the page must show the session that has closed.
    price_budget = fb.get("price_sessions", 0)
    price_lag = None
    if price_asof:
        price_lag = sessions_behind(dt.date.fromisoformat(price_asof[:10]),
                                    last_completed_session(now_utc))
    add("Price / NAV (live_track)", price_asof, price_lag, price_budget,
        live.get("computed_at_utc"), basis="NYSE sessions", warn_at=price_budget)
    add("Breadth / regime panel (risk_overlay)", regime_asof, _bday_lag(regime_asof, run_date),
        fb["regime_bdays"], overlay.get("computed_at_utc"))
    add("Strategy equity (multi_strategy)", multi.get("common_end"),
        _bday_lag(multi.get("common_end"), run_date), fb.get("strategy_bdays", fb["regime_bdays"]),
        multi.get("computed_at_utc"))

    # Benchmark vs the live NAV (2026-09-06). The S&P curve's base is the
    # engine's committed export, extended by yfinance returns; when the vendor
    # withholds the newest bar the curve stops a session short of the NAV and
    # every benchmark-relative figure on the page mixes dates (the 2026-09-05
    # build printed S&P YTD +13.1% on Thursday's bar against Friday's mark).
    # Judged in NYSE sessions behind the NAV's OWN as-of, not the calendar: a
    # NAV that is itself late is the Price feed's finding, not this row's.
    bench_budget = fb.get("benchmark_sessions", 1)
    for key, cfg in registry.get("benchmarks", {}).items():
        bm = (benchmarks or {}).get(key)
        if cfg.get("type") != "engine" or not bm or not bm.get("asOf") or not price_asof:
            continue
        b_lag = sessions_behind(dt.date.fromisoformat(bm["asOf"][:10]),
                                dt.date.fromisoformat(price_asof[:10]))
        add(f"Benchmark {cfg.get('label', key)} (engine export)", bm["asOf"], b_lag,
            bench_budget, bm.get("export_written_at_utc"), basis="NYSE sessions",
            warn_at=0, versus=price_asof, source=bm.get("source"))

    messages = []

    # Name every feed that breaches or nears its budget. The banner renders
    # this list verbatim, and an empty list once demoted a hard STALE to the
    # generic "approaching its freshness budget" fallback (2026-08-25).
    for f in feeds:
        if f["level"] == "ok":
            continue
        lag, unit = f["bday_lag"], f["basis"]
        if lag is None:
            messages.append(f"{f['feed']}: as-of date missing.")
        elif f.get("versus"):
            messages.append(
                f"{f['feed']} is {lag} NYSE session{'' if lag == 1 else 's'} behind the live NAV "
                f"(benchmark as of {f['asOf']}, NAV as of {f['versus']}); benchmark-relative "
                f"figures mix dates until the vendor serves the missing bar.")
        elif f["level"] == "stale":
            messages.append(
                f"{f['feed']} is {lag} {unit[:-1] if lag == 1 else unit} behind "
                f"(as of {f['asOf']}; budget {f['budget_bdays']}).")
        else:
            messages.append(
                f"{f['feed']} is approaching its freshness budget "
                f"({lag} of {f['budget_bdays']} {unit}; as of {f['asOf']}).")

    # Consistency: current_state_since must equal the latest event date. A mismatch
    # is exactly how the stale-panel incident hid a regime change.
    events = overlay.get("events", [])
    consistency_ok = True
    if events:
        last_evt = events[-1].get("date")
        if last_evt != overlay.get("current_state_since"):
            consistency_ok = False
            messages.append(
                f"Regime since-date {overlay.get('current_state_since')} != latest event "
                f"{last_evt} — possible historical revision.")
    if overlay.get("historical_revision"):
        messages.append("Engine flagged a historical regime revision.")

    # Reconciliation of our recompute against the engine's own published figures.
    reconcile = stats.get("reconcile", {})
    if not reconcile.get("ok", True):
        messages.append("Recomputed stats diverge from engine figures beyond tolerance "
                        f"(diffs={reconcile.get('diffs')}).")

    if not bench_ok:
        messages.append(f"Benchmark feed unavailable: {bench_note}.")

    level = max((f["level"] for f in feeds), key=lambda lv: _RANK[lv])
    if not consistency_ok:
        level = "stale"
    elif not reconcile.get("ok", True) and _RANK[level] < _RANK["warn"]:
        level = "warn"

    return {
        "level": level,
        "ok": level == "ok",
        "feeds": feeds,
        "consistency_ok": consistency_ok,
        "reconcile_ok": bool(reconcile.get("ok", True)),
        "benchmark_ok": bool(bench_ok),
        "messages": messages,
        "source_commit": bundle.get("source_commit"),
        "checked_at": run_date.isoformat(),
    }
