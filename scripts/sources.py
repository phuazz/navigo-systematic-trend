"""Fetch the engine's raw outputs from the breadth-thrust-etf repository.

The engine (phuazz/breadth-thrust-etf) bakes its dashboard daily but does NOT
publish its data/*.json as standalone files on GitHub Pages, so we read them
from the repository's raw endpoint instead. Each file is the engine's own
output contract; we treat the fetched bundle as read-only upstream data.

Provenance: we also resolve the latest commit SHA on the source ref so the
monitor can stamp exactly which engine build produced the numbers it is showing.

For offline development, --local <path-to-breadth-thrust-etf> reads the files
straight off disk instead of the network.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

REQUEST_TIMEOUT = 30


def _fetch_url(url: str) -> bytes:
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.content


def fetch_commit_sha(api_base: str, ref: str) -> str | None:
    """Best-effort: latest commit SHA on the source ref (for provenance)."""
    try:
        url = f"{api_base}/commits/{ref}"
        data = json.loads(_fetch_url(url))
        return data.get("sha")
    except Exception as exc:  # provenance is non-fatal — never block the build on it
        print(f"  [sources] commit SHA lookup failed ({exc!r}); continuing", file=sys.stderr)
        return None


def load_sources(registry: dict, *, local: str | None = None) -> dict:
    """Return {filename: parsed_json, ...} plus a 'source_commit' key.

    Raises on any required file that cannot be loaded — a missing upstream feed
    must stop the build loudly rather than silently bake a partial dashboard.
    """
    src = registry["source"]
    files = src["files"]
    out: dict = {}

    if local:
        base = Path(local) / "data"
        print(f"  [sources] reading local: {base}")
        for fn in files:
            path = base / fn
            if not path.exists():
                raise FileNotFoundError(f"Local source missing: {path}")
            out[fn] = json.loads(path.read_text(encoding="utf-8"))
        out["source_commit"] = "LOCAL"
        return out

    raw_base = src["raw_base"]
    print(f"  [sources] fetching raw: {raw_base}")
    for fn in files:
        url = f"{raw_base}/{fn}"
        try:
            out[fn] = json.loads(_fetch_url(url))
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch required source {url}: {exc}") from exc
        print(f"    ok  {fn} ({len(json.dumps(out[fn]))//1024} KB)")

    out["source_commit"] = fetch_commit_sha(src["api_base"], src["ref"])
    return out


def _cli() -> None:
    from config import load_registry

    ap = argparse.ArgumentParser(description="Fetch engine sources (debug).")
    ap.add_argument("--portfolio", default="navigo-systematic-trend")
    ap.add_argument("--local", default=None, help="path to a local breadth-thrust-etf checkout")
    args = ap.parse_args()

    reg = load_registry(args.portfolio)
    bundle = load_sources(reg, local=args.local)
    print("commit:", bundle.get("source_commit"))
    for fn in reg["source"]["files"]:
        d = bundle[fn]
        print(fn, "->", d.get("computed_at_utc", "?"))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    _cli()
