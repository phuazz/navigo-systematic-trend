"""The fail-loud gates. These encode the engine's de-risk staleness incident as
regressions: a stale breadth panel must downgrade to 'stale', and a regime
since-date that disagrees with the latest event must be caught.
"""
import datetime as dt

from config import load_registry
import validate

REG = load_registry("navigo-systematic-trend")
RUN = dt.date(2026, 6, 20)


def _bundle(panel_end="2026-06-12", since="2026-04-09", last_event="2026-04-09",
            price_asof="2026-06-18"):
    return {
        "source_commit": "abc123",
        "live_track.json": {
            "computed_at_utc": "2026-06-19T22:50:00Z",
            "live_dates": [price_asof], "anchor_date": "2026-06-17",
            "regime_state": "RISK_ON",
        },
        "risk_overlay.json": {
            "computed_at_utc": "2026-06-20T02:00:00Z",
            "panel_end_date": panel_end,
            "current_state": "RISK_ON", "current_state_since": since,
            "events": [{"date": "2026-03-27", "direction": "RISK_OFF", "breadth": 0.199},
                       {"date": last_event, "direction": "RISK_ON", "breadth": 0.502}],
            "historical_revision": [],
        },
        "multi_strategy.json": {"computed_at_utc": "2026-06-19T22:50:00Z", "common_end": "2026-06-17"},
    }


_OK_STATS = {"reconcile": {"ok": True, "diffs": {}}}


def test_fresh_feeds_pass():
    h = validate.run(_bundle(), REG, RUN, _OK_STATS, bench_ok=True, bench_note="ok")
    assert h["level"] == "ok" and h["ok"] and h["consistency_ok"]


def test_stale_breadth_panel_downgrades():
    # Panel three weeks old, well beyond the regime budget -> stale.
    h = validate.run(_bundle(panel_end="2026-05-20"), REG, RUN, _OK_STATS,
                     bench_ok=True, bench_note="ok")
    assert h["level"] == "stale"
    regime_feed = next(f for f in h["feeds"] if "regime" in f["feed"])
    assert regime_feed["level"] == "stale"


def test_since_date_event_mismatch_flagged():
    # since-date disagrees with the latest event -> the exact incident signature.
    h = validate.run(_bundle(since="2026-05-02"), REG, RUN, _OK_STATS,
                     bench_ok=True, bench_note="ok")
    assert not h["consistency_ok"]
    assert h["level"] == "stale"
    assert any("since-date" in m for m in h["messages"])


def test_reconcile_failure_warns():
    bad = {"reconcile": {"ok": False, "diffs": {"sharpe": 0.4}}}
    h = validate.run(_bundle(), REG, RUN, bad, bench_ok=True, bench_note="ok")
    assert h["level"] in ("warn", "stale") and not h["reconcile_ok"]


def test_benchmark_unavailable_noted():
    h = validate.run(_bundle(), REG, RUN, _OK_STATS, bench_ok=False, bench_note="yfinance down")
    assert not h["benchmark_ok"]
    assert any("Benchmark" in m for m in h["messages"])
