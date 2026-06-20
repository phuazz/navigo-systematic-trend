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
"""
from __future__ import annotations

import datetime as dt

import numpy as np

_RANK = {"ok": 0, "warn": 1, "stale": 2}


def _bday_lag(asof: str | None, run_date: dt.date) -> int | None:
    if not asof:
        return None
    a = dt.date.fromisoformat(asof[:10])
    return int(np.busday_count(a, run_date))


def run(bundle: dict, registry: dict, run_date: dt.date, stats: dict,
        bench_ok: bool, bench_note: str) -> dict:
    fb = registry["freshness"]
    live = bundle["live_track.json"]
    overlay = bundle["risk_overlay.json"]
    multi = bundle["multi_strategy.json"]

    price_asof = (live.get("live_dates") or [None])[-1] or live.get("anchor_date")
    regime_asof = overlay.get("panel_end_date")

    feeds = []

    def add(name, asof, lag, budget, computed):
        level = "ok"
        if lag is None:
            level = "warn"
        elif lag > budget:
            level = "stale"
        elif lag > budget - 2:
            level = "warn"
        feeds.append({"feed": name, "asOf": asof, "bday_lag": lag, "budget_bdays": budget,
                      "level": level, "computed_at": computed})

    add("Price / NAV (live_track)", price_asof, _bday_lag(price_asof, run_date),
        fb["price_bdays"], live.get("computed_at_utc"))
    add("Breadth / regime panel (risk_overlay)", regime_asof, _bday_lag(regime_asof, run_date),
        fb["regime_bdays"], overlay.get("computed_at_utc"))
    add("Strategy equity (multi_strategy)", multi.get("common_end"),
        _bday_lag(multi.get("common_end"), run_date), fb["regime_bdays"],
        multi.get("computed_at_utc"))

    messages = []

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
