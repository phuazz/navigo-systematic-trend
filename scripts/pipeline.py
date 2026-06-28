"""Build the monitor: fetch -> adapt -> compute -> validate -> write -> bake.

Run:  python scripts/pipeline.py [--local PATH] [--no-benchmarks]

Produces docs/data/portfolio-<id>.json (the client fetches this) and bakes
template.html -> docs/index.html. The dataset is the contract; the HTML is a
thin renderer, so the bake is a copy (no server-side templating needed).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adapter  # noqa: E402
import metrics  # noqa: E402
import prices as prices_mod  # noqa: E402
import validate  # noqa: E402
import valuation  # noqa: E402
from benchmarks import build_benchmarks  # noqa: E402
from config import (ACTIVE_PORTFOLIO_IDS, DATA_DIR, DOCS_DATA_DIR, DOCS_INDEX,  # noqa: E402
                    TEMPLATE, VALUATION_LAYER_ENABLED, dataset_path, load_registry)
from sources import load_sources  # noqa: E402


def build_dataset(portfolio_id: str, *, local: str | None = None,
                  run_date: dt.date | None = None, use_benchmarks: bool = True) -> dict:
    run_date = run_date or dt.datetime.now(dt.timezone.utc).date()
    reg = load_registry(portfolio_id)
    print(f"[{portfolio_id}] building (run_date={run_date})")

    bundle = load_sources(reg, local=local)
    live = bundle["live_track.json"]
    multi = bundle["multi_strategy.json"]
    overlay = bundle["risk_overlay.json"]

    # Weights first (drives which holdings need price coverage).
    weights = adapter.build_weights(live, reg)
    price_series, price_meta = prices_mod.build_prices(
        bundle["holdings_prices_1y.json"], weights["rows"], reg, fetch_missing=use_benchmarks)

    # Benchmarks aligned to the deployed model's date axis.
    model_dates = overlay["gated_variants"][reg["source"]["deployed_key"]]["dates"]
    if use_benchmarks:
        benchmarks, bench_ok, bench_note = build_benchmarks(model_dates, reg, live.get("live_dates"))
    else:
        benchmarks, bench_ok, bench_note = {}, False, "skipped (--no-benchmarks)"
    print(f"  benchmarks: ok={bench_ok} ({bench_note})")

    equity, model_bt, model_full = adapter.build_equity(live, multi, overlay, reg, benchmarks)
    stats = adapter.build_stats(model_bt, model_full, overlay, reg, benchmarks)
    regime = adapter.build_regime(overlay, live)
    risk = adapter.build_risk(price_series, weights, reg)
    attribution = adapter.build_attribution(equity, weights, price_series, reg,
                                            risk["by_ticker"], ref_date=model_full.index[-1])
    pnl = adapter.build_pnl(model_full, benchmarks)
    holdings_prices = adapter.build_holdings_prices(bundle["holdings_prices_1y.json"], price_series, weights)
    signals = adapter.build_signals(weights, price_meta, reg)
    monthly = metrics.monthly_matrix(model_full)

    prev = _load_prev(portfolio_id)
    changes = adapter.build_changes(weights, prev)

    # Append-only rebalance/trade ledger (persists across daily builds).
    ledger = _load_ledger(portfolio_id)
    new_ledger, fwd_trades = adapter.build_trades(ledger, weights, reg, stats["end"])
    _save_ledger(portfolio_id, new_ledger)                      # keep forward ledger as a fallback
    hist = adapter.build_weight_history(bundle, reg, overlay)   # full reconstruction (None if sources missing)
    trades = hist or fwd_trades
    trades["actions"] = adapter.build_action_history(overlay)   # de-risk + EM-tilt history since inception
    print(f"  trades: {'reconstructed full history' if hist else 'forward ledger'} — "
          f"{trades.get('count')} rebalances since {trades.get('since')}, {len(trades.get('actions', []))} actions")

    health = validate.run(bundle, reg, run_date, stats, bench_ok, bench_note)

    meta = {
        "id": reg["id"], "name": reg["name"], "descriptor": reg["descriptor"],
        "status": reg["status"], "inception": reg["inception"], "rebalance": reg["rebalance"],
        "oos_start": reg.get("oos_start"), "in_sample_end": reg.get("in_sample_end"),
        "base_currency": reg["base_currency"], "cost_assumption_bps": reg["cost_assumption_bps"],
        "asOf": stats["end"], "live_asOf": (equity["live"] or {}).get("dates", [stats["end"]])[-1]
        if equity.get("live") else stats["end"],
        "built_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_commit": bundle.get("source_commit"),
        "source_repo": reg["source"]["repo"],
        "engine_computed_at": {f: bundle[f].get("computed_at_utc") for f in reg["source"]["files"]},
        "health_level": health["level"],
    }

    dataset = {
        "meta": meta, "weights": weights, "equity": equity, "stats": stats,
        "regime": regime, "attribution": attribution, "risk": risk, "pnl": pnl,
        "holdings_prices": holdings_prices, "signals": signals, "trades": trades,
        "monthly": monthly, "changes": changes, "health": health,
    }

    # Valuation layer (DESIGN.md Phase 1), default OFF. When enabled, compute
    # Navigo's own daily mark and reconcile it against the engine's live_equity,
    # attaching it under a distinct key. The dataset is otherwise untouched, so
    # the off-path production output is byte-for-byte identical to the renderer.
    if VALUATION_LAYER_ENABLED:
        val = valuation.build(live, reg)
        if val:
            dataset["valuation"] = val
            _report_valuation(val)

    _report(dataset)
    return dataset


def _load_prev(portfolio_id: str) -> dict | None:
    p = dataset_path(portfolio_id, docs=True)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _ledger_path(portfolio_id: str, *, docs: bool) -> Path:
    base = DOCS_DATA_DIR if docs else DATA_DIR
    return base / f"rebalance-{portfolio_id}.json"


def _load_ledger(portfolio_id: str) -> dict | None:
    p = _ledger_path(portfolio_id, docs=True)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save_ledger(portfolio_id: str, ledger: dict) -> None:
    for docs in (False, True):
        p = _ledger_path(portfolio_id, docs=docs)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(ledger, separators=(",", ":")), encoding="utf-8")


def _report_valuation(val: dict) -> None:
    cov = val["coverage"]
    rec = val.get("reconcile")
    print(f"  valuation: weights_as_of={val['weights_as_of']} nav_as_of={val['nav_as_of']} "
          f"(age {val['weights_age_bdays']} bdays); coverage {cov['covered_weight']:.4f} "
          f"of 1.0, {'complete' if cov['complete'] else 'INCOMPLETE: ' + ','.join(cov['uncovered'])}")
    if rec:
        s, a = rec["settled"], rec["all"]
        print(f"  valuation reconcile vs engine live_equity: settled max={s['max_abs_bps']}bps "
              f"mean={s['mean_abs_bps']}bps (n={s['n']}); incl run-day max={a['max_abs_bps']}bps (n={a['n']})")


def _report(ds: dict) -> None:
    s, h, r = ds["stats"], ds["health"], ds["regime"]
    print(f"  stats: sharpe={s['sharpe']} cagr={s['cagr']} maxDD={s['max_dd']} "
          f"YTD={s['period_returns']['YTD']}")
    print(f"  reconcile ok={s['reconcile']['ok']} diffs={s['reconcile']['diffs']}")
    print(f"  regime: {r['state']} since {r['since']} breadth={r['breadth']} "
          f"EEM tilt={r['eem_tilt']['state']}")
    print(f"  health: {h['level'].upper()}  {('; '.join(h['messages']) or 'all feeds fresh')}")


def write_dataset(portfolio_id: str, dataset: dict) -> None:
    for docs in (False, True):
        p = dataset_path(portfolio_id, docs=docs)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dataset, separators=(",", ":")), encoding="utf-8")
    size = dataset_path(portfolio_id, docs=True).stat().st_size
    print(f"  wrote dataset ({size // 1024} KB)")


def bake_template() -> None:
    if not TEMPLATE.exists():
        print("  [bake] template.html not present yet — skipping HTML bake")
        return
    DOCS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATE, DOCS_INDEX)
    print(f"  baked {TEMPLATE.name} -> {DOCS_INDEX.relative_to(TEMPLATE.parent)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the Navigo model-portfolio monitor.")
    ap.add_argument("--local", default=None, help="path to a local breadth-thrust-etf checkout")
    ap.add_argument("--no-benchmarks", action="store_true", help="skip yfinance (fast offline build)")
    ap.add_argument("--portfolio", default=None, help="build only this portfolio id")
    args = ap.parse_args()

    ids = [args.portfolio] if args.portfolio else ACTIVE_PORTFOLIO_IDS
    for pid in ids:
        ds = build_dataset(pid, local=args.local, use_benchmarks=not args.no_benchmarks)
        write_dataset(pid, ds)
    bake_template()
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
