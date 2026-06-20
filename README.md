# Navigo Systematic Trend — Model Portfolio Monitor

A real-time monitoring dashboard for Navigo Investment Management's first systematic
**model (paper) portfolio**: *Navigo Systematic Trend* — a breadth- and trend-driven
global ETF rotation with a systematic de-risk overlay. Published to GitHub Pages.

> **Paper model.** This is a research and monitoring tool for a hypothetical model
> portfolio. It is not investment advice, an offer, or a record of actual trading.
> See the Methodology tab and the disclaimer footer.

**Live dashboard:** https://phuazz.github.io/navigo-systematic-trend/

## What it monitors

The deployed model is the `blend_35_35_10_20_gated_eem_tilted` strategy produced daily by
the upstream [breadth-thrust-etf](https://github.com/phuazz/breadth-thrust-etf) engine:

| Sleeve | Target | Mechanism |
|--------|-------:|-----------|
| A — US Sector Breadth | 35% | US sector top-K breadth rotation (K=7) |
| B — Cross-Asset Trend | 35% | Trend rotation across broad-market / regional / commodity sleeves (200-day MA) |
| C — Thematic Momentum | 10% | Cyber, clean energy, solar, battery, China-semis momentum |
| D — Europe Sector Breadth | 20% | Stoxx 600 sector breadth rotation |
| Overlay — De-risk gate | — | Below 20% S&P 500 breadth, shift 50% of NAV to SHY; re-engage above 50% |
| Overlay — EM tilt | +10% | EEM/SPY golden cross tilts 10% NAV to EEM, funded from sleeve B (weak evidence) |

Deployed backtest (engine figures): Sharpe ≈ 1.28, CAGR ≈ 15.4%, max drawdown ≈ −16.3%,
since 8 Nov 2018. The de-risk gate roughly halves drawdown versus the ungated blend.

## Architecture

This repo is a **thin, robust consumer**. It never re-runs the strategy; each day it
fetches the engine's published outputs, normalises them into its own data contract,
recomputes presentation-grade analytics, validates freshness, and bakes the dashboard.

```
breadth-thrust-etf (engine)            navigo-systematic-trend (this repo)
  data/live_track.json        ──┐        scripts/sources.py   fetch raw @main + commit SHA
  data/multi_strategy.json    ──┼──▶      scripts/adapter.py   normalise → data contract
  data/risk_overlay.json      ──┤        scripts/metrics.py   stats / attribution / monthly
  data/holdings_prices_1y.json──┘        scripts/benchmarks.py SPY + 60/40 via yfinance
                                          scripts/prices.py    holdings price panel (+yf supplement)
                                          scripts/validate.py  fail-loud freshness / consistency
                                          scripts/pipeline.py  orchestrate → docs/

  docs/data/portfolio-navigo-systematic-trend.json   ← the client fetches this
  docs/index.html                                     ← baked from template.html
```

Source data is fetched from the engine's `main` branch via `raw.githubusercontent.com`
(the engine is public; it does not publish its `data/*.json` to Pages directly).

### Multi-portfolio by design

Each portfolio is one registry file under `portfolios/<id>.json` (config, sleeve
allocations, benchmarks, freshness budgets, ETF metadata). Adding a second strategy is a
new registry file (and an adapter if its source shape differs) — not a restructure.
`scripts/config.py:ACTIVE_PORTFOLIO_IDS` lists what is built.

## Dashboard tabs

**Overview** · **Allocation** · **Performance** · **Attribution** · **Risk & Regime** ·
**Signals** · **Data Health** · **Methodology** — built so a performance analyst, a CIO,
and a quant PM each find what they look for: benchmark-relative performance and capture;
exposures, concentration and the regime state; sleeve/ETF attribution, a risk/return scatter
and risk-contribution decomposition, correlation and signal transparency; and first-class
data-integrity surfacing. Allocation rows are tap-to-expand, revealing a per-holding price
chart (close + 50/200-day MA) and trend signals. Short-horizon P&L (1-day / 1-week / 1-month
vs benchmarks) is calendar-weekday anchored, so an uneven multi-market calendar reads correctly.

## Build & develop

```bash
pip install -r requirements.txt

python scripts/pipeline.py                 # fetch from engine @main, full build
python scripts/pipeline.py --local ../breadth-thrust-etf   # read engine data off disk
python scripts/pipeline.py --no-benchmarks # skip yfinance (fast offline build)

python -m pytest tests/ -q                 # adapter, metrics, date-boundary, staleness gates

npx serve docs                             # preview the built dashboard
```

`template.html` is the editable source (fetch-based, works standalone for dev). The build
copies it to `docs/index.html` and writes the dataset to `docs/data/`. Never edit
`docs/index.html` directly.

## Automation

`.github/workflows/daily_monitor.yml` runs Mon–Fri 22:10 UTC (≈40 min after the engine's
21:30 UTC daily refresh): fetch → build → validate → test → commit `docs/` → Pages.

## Data integrity

The monitor surfaces, rather than hides, the failure modes that matter — the direct lesson
of an upstream incident where a confident regime state was published on an 11-week-stale
breadth panel. Every build:

- checks each feed's business-day lag against a budget (STALE banner + red Data Health on breach);
- asserts the regime since-date equals the latest switch event (the incident's signature);
- reconciles its recomputed Sharpe/CAGR/max-DD against the engine's own figures;
- renders the live mark-to-market extension as a distinct dashed segment — never silently spliced;
- stamps the engine commit SHA and every feed's `computed_at` for provenance.

## Related

- [breadth-thrust-etf](https://github.com/phuazz/breadth-thrust-etf) — the strategy engine (upstream source).
