# Navigo Systematic Trend — Working Notes

Durable context for future sessions on this repository. Layers on the vault `CLAUDE.md`.

## What this project is

A monitoring dashboard for Navigo's first systematic **model (paper) portfolio**. It is a
**read-only consumer** of the [breadth-thrust-etf](https://github.com/phuazz/breadth-thrust-etf)
engine: it fetches that engine's daily JSON outputs, normalises them, recomputes analytics,
and bakes a Portfolio-Command-Centre-styled dashboard to GitHub Pages. It never re-runs the
strategy. Context is Navigo (the fund), not CGSI or Personal.

## Hard rules

- **Never re-run or re-tune the strategy here.** Signal logic, weights and regime live in the
  engine. This repo only presents what the engine publishes.
- **Never modify the breadth-thrust-etf repo from a session in this repo.** It is upstream.
- **`template.html` is the source dashboard; never edit `docs/index.html`.** The latter is baked.
- **`docs/data/*.json` is generated** by the pipeline — never hand-edit. The single manual
  source of portfolio config is `portfolios/<id>.json`.
- **Dates via libraries only** — Python `datetime`/`dateutil` in the pipeline, `date-fns`-style
  care in the browser (`pct`/period anchors are computed in Python, which is unit-tested).

## Build

```
python scripts/pipeline.py                 # production: fetch engine @main + yfinance benchmarks
python scripts/pipeline.py --local ../breadth-thrust-etf   # offline source
python scripts/pipeline.py --no-benchmarks # skip yfinance (fastest)
python -m pytest tests/ -q                 # must stay green
npx serve docs                             # preview
```

## File sizes

`template.html` is ~70 KB and safe to read. `docs/index.html` is a baked copy of it.
`docs/data/portfolio-*.json` is ~450 KB (full equity histories + per-holding price panels for
the expandable Allocation charts) — read structure, not blindly.

## Data integrity philosophy

This dashboard exists partly because the engine once published a confident regime state on a
stale breadth panel. So `scripts/validate.py` runs fail-loud freshness, regime-consistency and
statistics-reconciliation gates on every build, and the **Data Health** tab makes them visible.
Preserve this: any new feed gets a freshness budget and a per-feed as-of on Data Health. The
live mark-to-market extension must always render as a distinct segment, never spliced into the
backtest curve silently.

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
