# Multi-Strategy Portfolio — Model Portfolio Monitor

A real-time monitoring dashboard for a personal systematic
**model (paper) portfolio**: *Multi-Strategy Portfolio* — a breadth- and trend-driven
global ETF rotation with a systematic de-risk overlay. Published to GitHub Pages.

> **Paper model.** This is a research and monitoring tool for a hypothetical model
> portfolio. It is not investment advice, an offer, or a record of actual trading.
> See the Methodology tab and the disclaimer footer.

**Live dashboard:** https://phuazz.github.io/multi-strategy-portfolio/

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

Headline statistics live on the dashboard's KPI strip — recomputed on every build and
reconciled against the engine's published figures, with the Sharpe convention and error
bar named on the tile. This file deliberately does not restate them: three engine
restatements in one week of August 2026 (WS10 holiday cadence, WS11 survivorship, WS16
cross-panel closure) each left a hardcoded copy stale on arrival. Backtest window since
8 Nov 2018; the de-risk gate roughly halves max drawdown versus the ungated blend.

## Architecture

This repo is a **thin, robust consumer**. It never re-runs the strategy; each day it
fetches the engine's published outputs, normalises them into its own data contract,
recomputes presentation-grade analytics, validates freshness, and bakes the dashboard.

```
breadth-thrust-etf (engine)            multi-strategy-portfolio (this repo)
  data/live_track.json        ──┐        scripts/sources.py   fetch raw @main + commit SHA
  data/multi_strategy.json    ──┼──▶      scripts/adapter.py   normalise → data contract
  data/risk_overlay.json      ──┤        scripts/metrics.py   stats / attribution / monthly
  data/holdings_prices_1y.json──┤        scripts/benchmarks.py S&P = engine export + yf extension; others via yfinance
  data/benchmark_spy.json     ──┘        scripts/prices.py    holdings price panel (+yf supplement)
                                          scripts/validate.py  fail-loud freshness / consistency
                                          scripts/pipeline.py  orchestrate → docs/

  docs/data/portfolio-multi-strategy-portfolio.json   ← the client fetches this
  docs/index.html                                     ← baked from template.html
```

Source data is fetched from the engine's `main` branch via `raw.githubusercontent.com`
(the engine is public; it does not publish its `data/*.json` to Pages directly).

**The S&P 500 benchmark is the engine's series (2026-09-06).** `data/benchmark_spy.json` —
the engine's committed export of adjusted SPY closes, the series behind its weekly email and
factsheet — is the base of this page's S&P curve; yfinance only chains daily returns after
the export's last date (the export refreshes with the engine's local Tue/Wed/Sat/Sun runs,
so mid-week the extension is one to three sessions). Both surfaces therefore carry one S&P
figure by construction, and the page reconciles to the email, never the reverse. No
benchmark is forward-filled onto a date it has no bar for: on 2026-09-05 the 01:22 UTC
fetch received no Friday SPY bar, the old alignment carried Thursday's close onto Friday's
date, and the page printed S&P YTD +13.1% against the model's Friday mark while the email
said +12.7%. A benchmark now ends at its own last served close, carries `asOf`, and Data
Health scores it in NYSE sessions behind the NAV (one behind = WARN, with the date caveat
beside every S&P-relative figure). The model side was never the problem — its series was
byte-identical to the engine's — but the engine restates the deployed history in its
weekend run, so the Sunday build below keeps the page on the same vintage as the email.

### Multi-portfolio by design

Each portfolio is one registry file under `portfolios/<id>.json` (config, sleeve
allocations, benchmarks, freshness budgets, ETF metadata). Adding a second strategy is a
new registry file (and an adapter if its source shape differs) — not a restructure.
`scripts/config.py:ACTIVE_PORTFOLIO_IDS` lists what is built.

## Dashboard tabs

Above the tabs, a **headline synthesis line** states in one sentence what the KPI strip's
eight facts add up to: the model's YTD return, how far ahead of or behind each house
comparator it stands, and the computed risk state (gate, EM tilt, drawdown, cash) beneath.
Both comparators — the S&P 500 and a global 60/40 — are differenced against figures from the
same `metrics.period_returns` call over the same window, so the "pp ahead" arithmetic is
like-for-like; per-figure tooltips carry the basis and the as-at date. A US-equity-only anchor
is deliberately not used alone, as it would flatter or punish a global multi-asset rotation on
asset mix. No currency figure appears in the line: this is a paper model with no capital.

**Live session tiles.** The P&L strip opens with **US session** and **Europe session** rather than
one blended intraday figure. The book is ~78% US-listed and ~20% Xetra, so the two venues barely
overlap: a single number was part-stale during European hours, and gating it on half of total NAV
made it dark for the whole European session instead, since Xetra can never reach that share. Each
tile is the **NAV contribution** of its own venue — weight times return, so the two are additive —
and each is judged live only when at least half of *its own* sleeve is quoting inside 30 minutes.
Holdings that are not trading contribute nothing rather than contributing their last completed
session, the SPY comparator appears only when SPY itself is live (scaled to the sleeve's weight so
the comparison is like-for-like), and any holding on a third calendar is reported as a residual
rather than dropped. When a venue is closed its tile says so; the 1-Day tile carries the last
completed session for the whole book.

**Execution convention.** A rebalance dated Friday is assumed **filled at that Friday's close**,
not on the Monday. Each sleeve reads its signal at the session *before* the rebalance date
(`prev_idx = closes.index.get_loc(rd) - 1`, normally Thursday), stamps the target weights on the
Friday, and earns them through `weight_panel.shift(1)` against close-to-close returns. Because
`shift(1)` on close-to-close returns makes the new Friday weight earn the Friday-close to
Monday-close bar, the position must already exist at Friday's close — hence a Friday fill. The
signal-to-fill gap is therefore one full session (Thursday close to Friday close), long enough to
work a market-on-close order, and is not a weekend. The digest labels the rows
"priced at Fri … close" and states that they are model weights, never executed trades.

Overview also carries a **What changed** digest, windowed on the last rebalance rather than
the last day — the model rebalances weekly, so a daily window would be empty most days. When
that rebalance falls in the quietest decile the card says so and shows the last substantive
one (at or above the median) beneath it, so it is never dead and never inflates a quiet week.
Both cut-offs are percentiles recomputed from the full history at run time. Overlay switches
are rare and material, so one firing at or after the previous rebalance is flagged rather than
listed. Turnover counts every move while the rows count only moves ≥0.5pp (the blotter's
threshold), so the row count is labelled with that threshold — the two measures differ and the
Trades tab reports the unfiltered count for the same date. The digest and the Trades tab share
`blotterAll()` / `latestRebalance()`, so they cannot drift apart on what "latest" means.

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

`.github/workflows/daily_monitor.yml` runs Mon–Fri 23:40 UTC: fetch → build → validate →
test → capture-integrity check → commit `docs/` → Pages. The cron sits past the engine's
*measured* publish tail (its 21:30 UTC daily has landed as late as 23:18 UTC) rather than
its scheduled time — fetching before the engine publishes would bake yesterday's data as
latest inside the freshness budget, silently. Publishes follow the engine's cadence rule:
every Friday after the US close even on US market holidays, using the latest populated
close.

**Ops alerting (2026-07-03)**: the daily workflow emails the operator (GMAIL_USER) on any
failure, and on a capture warning (baked live as-of behind the NYSE calendar, or baked
health not `ok`) while still publishing — staleness is surfaced, never hidden. Outside-in,
`.github/workflows/sentinel.yml` (daily 05:05 UTC = 13:05 SGT, sized to GitHub's
cron-delay tail) fetches the DEPLOYED dataset and emails `[SENTINEL]` on session lag or a
stale health level — it shares no state with the build, so it catches a green run that
published wrong artefacts. Requires the `GMAIL_USER` / `GMAIL_APP_PASSWORD` repository
secrets. `VERIFY_DASHBOARD.md` holds the manual deep-audit prompt.

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
- [DESIGN.md](DESIGN.md) — target architecture for the monitor as a multi-strategy valuation layer (engines generate, the monitor values), and the contract each engine must publish. Proposal; the thin-renderer path above remains production until it lands.
