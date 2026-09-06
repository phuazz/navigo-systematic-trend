"""The fail-loud gates. These encode the engine's de-risk staleness incident as
regressions: a stale breadth panel must downgrade to 'stale', and a regime
since-date that disagrees with the latest event must be caught.

The live-NAV block additionally encodes the 2026-08-21 miss, where the engine's
own guard blocked publication, the monitor baked Thursday's mark on a Friday,
and the page still read 'ok' on a 4-business-day budget. Python datetime months
are 1-indexed (January = 1).
"""
import datetime as dt

from config import load_registry
import validate

REG = load_registry("multi-strategy-portfolio")
RUN = dt.date(2026, 6, 20)
# 2026-06-19 is Juneteenth, an NYSE holiday, so the last completed session at
# this instant is Thursday 2026-06-18 — the default bundle's price as-of.
NOW = dt.datetime(2026, 6, 19, 23, 55, tzinfo=dt.timezone.utc)


def _price_feed(health):
    return next(f for f in health["feeds"] if "Price" in f["feed"])


def _live_level(price_asof, now_utc, run_date=RUN):
    """Level of the live-NAV feed alone, judged at a given instant."""
    h = validate.run(_bundle(price_asof=price_asof), REG, run_date, _OK_STATS,
                     bench_ok=True, bench_note="ok", now_utc=now_utc)
    return _price_feed(h)


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
    h = validate.run(_bundle(), REG, RUN, _OK_STATS, bench_ok=True, bench_note="ok",
                     now_utc=NOW)
    assert h["level"] == "ok" and h["ok"] and h["consistency_ok"]


def test_stale_breadth_panel_downgrades():
    # Panel three weeks old, well beyond the regime budget -> stale.
    h = validate.run(_bundle(panel_end="2026-05-20"), REG, RUN, _OK_STATS,
                     bench_ok=True, bench_note="ok", now_utc=NOW)
    assert h["level"] == "stale"
    regime_feed = next(f for f in h["feeds"] if "regime" in f["feed"])
    assert regime_feed["level"] == "stale"


def test_since_date_event_mismatch_flagged():
    # since-date disagrees with the latest event -> the exact incident signature.
    h = validate.run(_bundle(since="2026-05-02"), REG, RUN, _OK_STATS,
                     bench_ok=True, bench_note="ok", now_utc=NOW)
    assert not h["consistency_ok"]
    assert h["level"] == "stale"
    assert any("since-date" in m for m in h["messages"])


def test_reconcile_failure_warns():
    bad = {"reconcile": {"ok": False, "diffs": {"sharpe": 0.4}}}
    h = validate.run(_bundle(), REG, RUN, bad, bench_ok=True, bench_note="ok",
                     now_utc=NOW)
    assert h["level"] in ("warn", "stale") and not h["reconcile_ok"]


def test_benchmark_unavailable_noted():
    h = validate.run(_bundle(), REG, RUN, _OK_STATS, bench_ok=False, bench_note="yfinance down",
                     now_utc=NOW)
    assert not h["benchmark_ok"]
    assert any("Benchmark" in m for m in h["messages"])


# --- live NAV: judged in NYSE sessions, not business days -------------------

def test_live_nav_one_session_behind_is_stale():
    """The 2026-08-21 regression. Thursday's mark baked on a Friday evening,
    after Friday's close: one session behind, and the page must say so. The old
    4-business-day budget scored this 'ok'."""
    feed = _live_level("2026-08-20",
                       dt.datetime(2026, 8, 21, 23, 55, tzinfo=dt.timezone.utc),
                       run_date=dt.date(2026, 8, 21))
    assert feed["bday_lag"] == 1
    assert feed["level"] == "stale"
    assert feed["basis"] == "NYSE sessions"


def test_live_nav_current_is_ok_when_cron_slips_past_midnight():
    """The false-alarm guard, and the reason a business-day budget cannot do
    this job: Monday's mark read on Tuesday 01:00 UTC scores the SAME
    business-day lag of 1 as the miss above, but it is perfectly healthy."""
    feed = _live_level("2026-08-24",
                       dt.datetime(2026, 8, 25, 1, 0, tzinfo=dt.timezone.utc),
                       run_date=dt.date(2026, 8, 25))
    assert feed["bday_lag"] == 0 and feed["level"] == "ok"


def test_live_nav_holiday_dated_dashboard_is_not_stale():
    """Thursday's mark on a Friday holiday is correct, not stale — the reason
    the check uses the true exchange calendar rather than weekday arithmetic.
    2026-06-19 is Juneteenth."""
    feed = _live_level("2026-06-18", NOW)
    assert feed["bday_lag"] == 0 and feed["level"] == "ok"


def test_live_nav_month_boundary():
    """Month boundary: 2026-06-30 (Tue) into 2026-07-01 (Wed)."""
    fresh = _live_level("2026-06-30",
                        dt.datetime(2026, 7, 1, 1, 0, tzinfo=dt.timezone.utc),
                        run_date=dt.date(2026, 7, 1))
    assert fresh["bday_lag"] == 0 and fresh["level"] == "ok"
    behind = _live_level("2026-06-30",
                         dt.datetime(2026, 7, 1, 23, 55, tzinfo=dt.timezone.utc),
                         run_date=dt.date(2026, 7, 1))
    assert behind["bday_lag"] == 1 and behind["level"] == "stale"


def test_live_nav_year_boundary():
    """Year boundary: 2026-12-31 (Thu) across New Year's Day 2027-01-01 (Fri,
    an NYSE holiday) to the next session, Monday 2027-01-04."""
    holiday = _live_level("2026-12-31",
                          dt.datetime(2027, 1, 1, 12, 0, tzinfo=dt.timezone.utc),
                          run_date=dt.date(2027, 1, 1))
    assert holiday["bday_lag"] == 0 and holiday["level"] == "ok"
    behind = _live_level("2026-12-31",
                         dt.datetime(2027, 1, 4, 23, 55, tzinfo=dt.timezone.utc),
                         run_date=dt.date(2027, 1, 4))
    assert behind["bday_lag"] == 1 and behind["level"] == "stale"


# --- a breached feed must be named in messages ------------------------------

def test_stale_feed_named_in_messages():
    """The 2026-08-25 banner regression: live_track was a hard STALE, but with
    messages empty the page fell back to 'a feed is approaching its freshness
    budget'. A breached feed must be named, with lag, basis and as-of date."""
    h = validate.run(_bundle(price_asof="2026-08-20"), REG, dt.date(2026, 8, 21),
                     _OK_STATS, bench_ok=True, bench_note="ok",
                     now_utc=dt.datetime(2026, 8, 21, 23, 55, tzinfo=dt.timezone.utc))
    assert h["level"] == "stale"
    assert any("live_track" in m and "1 NYSE session behind" in m
               and "2026-08-20" in m for m in h["messages"])


def test_warn_feed_message_says_approaching():
    # Panel 7 business days old against a budget of 8: inside the approach band.
    h = validate.run(_bundle(panel_end="2026-06-11"), REG, RUN, _OK_STATS,
                     bench_ok=True, bench_note="ok", now_utc=NOW)
    assert h["level"] == "warn"
    assert any("risk_overlay" in m and "approaching" in m for m in h["messages"])


def test_fresh_feeds_produce_no_messages():
    # The banner only renders when there is something to say; a healthy build
    # must not accumulate noise.
    h = validate.run(_bundle(), REG, RUN, _OK_STATS, bench_ok=True, bench_note="ok",
                     now_utc=NOW)
    assert h["messages"] == []


# --- benchmark vs the live NAV (2026-09-06) ---------------------------------

def _bench(asof, source="engine export benchmark_spy.json to 2026-06-17 + yfinance returns for 1 session"):
    return {"SPY": {"asOf": asof, "source": source, "dates": [], "equity": [],
                    "export_written_at_utc": "2026-06-14T02:00:00+00:00"}}


def _bench_feed(h):
    return next(f for f in h["feeds"] if f["feed"].startswith("Benchmark"))


def test_benchmark_current_with_the_nav_is_ok_and_silent():
    h = validate.run(_bundle(), REG, RUN, _OK_STATS, bench_ok=True, bench_note="ok",
                     now_utc=NOW, benchmarks=_bench("2026-06-18"))
    f = _bench_feed(h)
    assert f["bday_lag"] == 0 and f["level"] == "ok" and f["basis"] == "NYSE sessions"
    assert f["versus"] == "2026-06-18" and f["source"].startswith("engine export")
    assert f["computed_at"] == "2026-06-14T02:00:00+00:00"
    assert h["messages"] == [] and h["level"] == "ok"


def test_benchmark_one_session_behind_the_nav_warns_and_names_both_dates():
    """The 2026-09-05 build: the model marked on Friday 4 Sep, SPY's Friday bar
    was withheld, the old alignment forward-filled Thursday and the page
    compared the two without a word. One session behind is WARN, with both
    dates in the message."""
    h = validate.run(_bundle(), REG, RUN, _OK_STATS, bench_ok=True, bench_note="ok",
                     now_utc=NOW, benchmarks=_bench("2026-06-17"))
    f = _bench_feed(h)
    assert f["bday_lag"] == 1 and f["level"] == "warn"
    assert h["level"] == "warn"
    assert any("Benchmark S&P 500" in m and "1 NYSE session behind the live NAV" in m
               and "2026-06-17" in m and "2026-06-18" in m for m in h["messages"])


def test_benchmark_two_sessions_behind_is_stale():
    h = validate.run(_bundle(), REG, RUN, _OK_STATS, bench_ok=True, bench_note="ok",
                     now_utc=NOW, benchmarks=_bench("2026-06-16"))
    assert _bench_feed(h)["level"] == "stale" and h["level"] == "stale"
    assert any("2 NYSE sessions behind" in m for m in h["messages"])


def test_benchmark_ahead_of_a_late_nav_is_the_price_feeds_finding():
    # NAV stuck on Wednesday, benchmark on Thursday: the benchmark row reads 0
    # (never negative) and the Price feed carries the staleness.
    h = validate.run(_bundle(price_asof="2026-06-17"), REG, RUN, _OK_STATS, bench_ok=True,
                     bench_note="ok", now_utc=NOW, benchmarks=_bench("2026-06-18"))
    assert _bench_feed(h)["bday_lag"] == 0 and _bench_feed(h)["level"] == "ok"
    assert _price_feed(h)["level"] == "stale"


def test_no_benchmark_row_without_an_engine_benchmark():
    h = validate.run(_bundle(), REG, RUN, _OK_STATS, bench_ok=True, bench_note="ok", now_utc=NOW)
    assert not any(f["feed"].startswith("Benchmark") for f in h["feeds"])
