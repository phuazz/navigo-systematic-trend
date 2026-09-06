# Multi-Strategy Portfolio — Working Notes

Durable context for future sessions on this repository. Layers on the vault `CLAUDE.md`.

## What this project is

A monitoring dashboard for a personal systematic **model (paper) portfolio**. It is a
**read-only consumer** of the [breadth-thrust-etf](https://github.com/phuazz/breadth-thrust-etf)
engine: it fetches that engine's daily JSON outputs, normalises them, recomputes analytics,
and bakes a Portfolio-Command-Centre-styled dashboard to GitHub Pages. It never re-runs the
strategy. Context is Personal (own book), not Navigo or CGSI. Renamed from navigo-systematic-trend on 2026-07-11; strategy, engine and contract unchanged.

## Hard rules

- **Never re-run or re-tune the strategy here.** Signal logic, weights and regime live in the
  engine. This repo only presents what the engine publishes.
- **Never modify the breadth-thrust-etf repo from a session in this repo.** It is upstream.
- **`template.html` is the source dashboard; never edit `docs/index.html`.** The latter is baked.
- **`docs/data/*.json` is generated** by the pipeline — never hand-edit. The single manual
  source of portfolio config is `portfolios/<id>.json`.
- **Dates via libraries only** — Python `datetime`/`dateutil` in the pipeline, `date-fns`-style
  care in the browser (`pct`/period anchors are computed in Python, which is unit-tested).
- **The S&P 500 benchmark is the engine's series.** `data/benchmark_spy.json` (the engine's
  committed export, the series behind its weekly email and factsheet) is the base of the SPY
  curve; yfinance only chains returns after its last date, and no benchmark is ever
  forward-filled onto a date it has no bar for (2026-09-05: a withheld Friday bar printed S&P
  YTD +13.1% against the email's +12.7%). Reconcile the page to the email, never the reverse.

## Build

```
python scripts/pipeline.py                 # production: fetch engine @main + yfinance benchmarks
python scripts/pipeline.py --local ../breadth-thrust-etf   # offline source
python scripts/pipeline.py --no-benchmarks # skip yfinance (fastest)
python -m pytest tests/ -q                 # must stay green
npx serve docs                             # preview
```

## File sizes

`template.html` is ~105 KB and safe to read. `docs/index.html` is a baked copy of it.
`docs/data/portfolio-*.json` is ~450 KB (full equity histories + per-holding price panels for
the expandable Allocation charts) — read structure, not blindly.

## Data integrity philosophy

This dashboard exists partly because the engine once published a confident regime state on a
stale breadth panel. So `scripts/validate.py` runs fail-loud freshness, regime-consistency and
statistics-reconciliation gates on every build, and the **Data Health** tab makes them visible.
Preserve this: any new feed gets a freshness budget and a per-feed as-of on Data Health. The
live mark-to-market extension must always render as a distinct segment, never spliced into the
backtest curve silently.

## Ops alerting and audit (2026-07-03)

Three layers, mirroring the engine repo: (1) `scripts/check_capture_integrity.py` runs in the
daily workflow — anchors the baked live as-of to the TRUE NYSE calendar
(`scripts/nyse_sessions.py`, pandas_market_calendars); a lag or non-ok health warns by email
but still publishes (surface, never hide); only a corrupt artefact blocks the commit. (2) Both
failure and warn emails go to GMAIL_USER — the `GMAIL_USER` / `GMAIL_APP_PASSWORD` repository
secrets must exist or the alert channel is dark. (3) `.github/workflows/sentinel.yml` (daily
05:05 UTC) checks the DEPLOYED dataset independently of the build. The daily cron is 23:40 UTC
deliberately — past the engine's measured publish tail, not its scheduled time; do not move it
earlier. `VERIFY_DASHBOARD.md` is the manual deep-audit prompt; run it for any "is the
dashboard fresh/working" question.

## Adding a portfolio

Drop `portfolios/<new-id>.json` (copy the existing one; adjust source keys, sleeves, benchmarks,
etf_meta), add the id to `ACTIVE_PORTFOLIO_IDS` in `scripts/config.py`, and — only if the new
source's JSON shape differs — branch in `adapter.py`. The dashboard becomes a multi-portfolio
selector when a second id is present.

## Commit discipline

Per vault: separate approvals for `git commit` and `git push`; British/Singapore English; no
contractions in commit messages, comments or docs. The daily workflow owns `docs/` — discard
local pipeline output (`git checkout -- docs/ data/`) before ending a session to keep the next
`git pull --rebase` clean.
