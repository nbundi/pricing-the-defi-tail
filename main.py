"""Data pipeline: pull DefiLlama hack ledger + protocol TVL histories,
normalize, and produce the inputs that ingest_consolidated.py and
risk_analysis.py consume.

Outputs:
    data/raw/defillama/hacks.csv  — per-hack normalised record from
                                    DefiLlama /hacks endpoint (input to
                                    ingest_consolidated.py)
    data/sector_tvl_panel.csv     — daily TVL by sector across all
                                    in-scope chains (input to
                                    risk_analysis.py for the bps
                                    denominator)

Cached intermediate data lands in ./data/raw/ (top-level API responses) and
./data/protocols/ (one JSON per protocol, transient — re-created on each
run). Pass --refresh to force re-fetch from DefiLlama.

This file no longer produces descriptive figures — the headline analysis is
in paper.tex and the figures it embeds are produced by risk_analysis.py.

Run:
    python main.py            # cached
    python main.py --refresh  # bypass cache
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROTO = DATA / "protocols"  # transient per-protocol JSON cache (re-created on run)
OUT = ROOT / "output"
for p in (RAW, PROTO, DATA, OUT):
    p.mkdir(parents=True, exist_ok=True)

# "As of" date for the trailing-365d headline table.
AS_OF = dt.date(2026, 5, 16)
WINDOW_DAYS = 365

# Broad sector buckets for the analysis.
SECTOR_BUCKETS: dict[str, set[str]] = {
    # Borrowing / lending markets — RWA Lending grouped here.
    "Lending":       {"Lending", "CDP", "RWA Lending"},
    "DEX":           {"Dexs", "DEX Aggregator"},
    "Bridge":        {"Bridge", "Cross Chain Bridge", "Chain"},
    # Derivatives now also absorbs the Liquid Staking / Restaking
    # buckets: at the protocol-event level they share the same OpRisk
    # signature (collateral-management failure, oracle dependency).
    "Derivatives":   {"Derivatives", "Liquid Staking", "Liquid Restaking"},
    "Yield":         {"Yield", "Yield Aggregator", "Farm"},
    "RWA":           {"RWA"},
    "Algo-Stables":  {"Algo-Stables", "Reserve Currency"},
}

API = "https://api.llama.fi"
UA = "Mozilla/5.0 (compatible; defi-exploit-loss-analysis/1.0)"

# Top-N protocols per sector to fetch full TVL history for.
# 50 typically captures 99%+ of category TVL on DefiLlama.
TOP_N_PER_SECTOR = 50


# %% HTTP helpers -----------------------------------------------------------

def http_stream_to_file(url: str, dest: Path, read_timeout: int = 300,
                       chunk: int = 65536) -> None:
    """Stream a URL to `dest`, writing chunks as they arrive. Writes to a
    .tmp sidecar and atomically renames on full success; a connection
    error mid-stream leaves the .tmp behind for inspection."""
    req = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(req, timeout=read_timeout) as r, open(tmp, "wb") as f:
                while True:
                    buf = r.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
            with open(tmp, "rb") as f:
                json.loads(f.read())
            tmp.replace(dest)
            return
        except Exception as e:
            last_err = e
            time.sleep(1 + attempt * 2)
    raise RuntimeError(f"stream failed for {url}: {last_err}")


def cache_get_json(url: str, cache_path: Path, refresh: bool = False) -> object:
    if cache_path.exists() and not refresh:
        try:
            return json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            pass
    http_stream_to_file(url, cache_path)
    return json.loads(cache_path.read_text())


# %% Step 1 — load base reference data --------------------------------------

def load_base(refresh: bool = False):
    hacks = cache_get_json(f"{API}/hacks", RAW / "hacks.json", refresh)
    protocols = cache_get_json(f"{API}/protocols", RAW / "protocols.json", refresh)
    chains = cache_get_json(f"{API}/v2/chains", RAW / "chains.json", refresh)
    return hacks, protocols, chains


def chain_universe(chains: list[dict]) -> set[str]:
    """All chains DefiLlama tracks."""
    return {c["name"] for c in chains}


# %% Step 2 — hack feature engineering --------------------------------------

def hacks_dataframe(hacks: list[dict], protocols: list[dict],
                    target_chains: set[str]) -> pd.DataFrame:
    """One row per hack with normalized fields and derived flags."""
    proto_by_id = {str(p.get("id")): p for p in protocols if p.get("id") is not None}

    def _norm(s: str) -> str:
        return "".join(ch.lower() for ch in (s or "") if ch.isalnum())
    proto_by_name: dict[str, dict] = {}
    for p in protocols:
        for key in (p.get("name"), p.get("slug")):
            if not key: continue
            proto_by_name.setdefault(_norm(key), p)

    HACK_NAME_OVERRIDES = {
        "Silo Finance": "silo-v2",
        "CrediX":       "credix",
    }
    HACK_CATEGORY_OVERRIDES: dict[str, str] = {}

    rows = []
    for h in hacks:
        amt = h.get("amount") or 0.0
        ret = h.get("returnedFunds") or 0.0
        net = max(0.0, amt - min(ret, amt))

        pid = h.get("defillamaId")
        proto = proto_by_id.get(str(pid)) if pid is not None else None
        if proto is None:
            slug_override = HACK_NAME_OVERRIDES.get(h.get("name") or "")
            if slug_override:
                proto = next((p for p in protocols if p.get("slug") == slug_override), None)
        if proto is None:
            proto = proto_by_name.get(_norm(h.get("name") or ""))
        category = (proto or {}).get("category")
        if category is None:
            category = HACK_CATEGORY_OVERRIDES.get(h.get("name") or "")

        chain_list = h.get("chain") or []
        in_scope_chain = any(c in target_chains for c in chain_list)

        sector = "Other"
        for bucket, cats in SECTOR_BUCKETS.items():
            if category in cats:
                sector = bucket
                break
        # Bridge override: trust the explicit flag in the hacks dataset.
        if h.get("bridgeHack"):
            sector = "Bridge"

        rows.append({
            "date": pd.to_datetime(h["date"], unit="s", utc=True).tz_convert(None).normalize(),
            "name": h["name"],
            "gross": float(amt),
            "recovered": float(min(ret, amt)),
            "net": float(net),
            "classification": h.get("classification"),
            "technique": h.get("technique"),
            "chains": ",".join(chain_list) if chain_list else "",
            "in_scope_chain": in_scope_chain,
            "bridge_hack": bool(h.get("bridgeHack")),
            "target_type": h.get("targetType"),
            "is_defi_protocol": h.get("targetType") == "DeFi Protocol",
            "defillamaId": h.get("defillamaId"),
            "category": category,
            "sector": sector,
        })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


# %% Step 3 — pick top-N protocols per sector to fetch history for ---------

def chain_tvl_snapshot(protocol_summary: dict, target_chains: set[str]) -> float:
    ct = protocol_summary.get("chainTvls") or {}
    total = 0.0
    for k, v in ct.items():
        if "-" in k:  # skip "Ethereum-borrowed", "Ethereum-staking" etc.
            continue
        if isinstance(v, (int, float)) and k in target_chains:
            total += v
    return total


def pick_top_protocols(protocols: list[dict], target_chains: set[str]) -> dict[str, list[dict]]:
    by_sector: dict[str, list[dict]] = {b: [] for b in SECTOR_BUCKETS}
    for p in protocols:
        cat = p.get("category")
        if not cat:
            continue
        for bucket, cats in SECTOR_BUCKETS.items():
            if cat in cats:
                by_sector[bucket].append(p)
                break
    out = {}
    for bucket, plist in by_sector.items():
        ranked = sorted(plist, key=lambda p: -chain_tvl_snapshot(p, target_chains))
        out[bucket] = ranked[:TOP_N_PER_SECTOR]
    return out


# %% Step 4 — fetch per-protocol historical TVL ----------------------------

def fetch_protocol(slug: str, refresh: bool = False) -> dict | None:
    fp = PROTO / f"{slug}.json"
    if fp.exists() and not refresh:
        try:
            return json.loads(fp.read_text())
        except json.JSONDecodeError:
            print(f"  ~ cache invalid for {slug}, re-fetching", file=sys.stderr)
    try:
        http_stream_to_file(f"{API}/protocol/{slug}", fp, read_timeout=300)
        return json.loads(fp.read_text())
    except Exception as e:
        print(f"  ! failed {slug}: {e}", file=sys.stderr)
        return None


def protocol_chain_tvl(p: dict, target_chains: set[str],
                       include_borrowed: bool = False) -> pd.Series:
    """Daily TVL contribution of a protocol restricted to `target_chains`.
    With `include_borrowed=True`, lending protocols' `<Chain>-borrowed`
    series are also summed in (gives supplied = idle + borrowed)."""
    ct = p.get("chainTvls") or {}
    series_parts = []
    for chain, payload in ct.items():
        if chain in ("borrowed",):
            continue
        base_chain = chain
        is_borrow = False
        if chain.endswith("-borrowed"):
            base_chain = chain[: -len("-borrowed")]
            is_borrow = True
        elif "-" in chain:
            continue
        if base_chain not in target_chains:
            continue
        if is_borrow and not include_borrowed:
            continue
        pts = (payload or {}).get("tvl") or []
        if not pts:
            continue
        s = pd.Series(
            [pt["totalLiquidityUSD"] for pt in pts],
            index=pd.to_datetime([pt["date"] for pt in pts], unit="s", utc=True).tz_convert(None).normalize(),
        )
        s = s.groupby(level=0).last()
        series_parts.append(s)
    if not series_parts:
        return pd.Series(dtype=float)
    df = pd.concat(series_parts, axis=1).ffill().fillna(0.0)
    return df.sum(axis=1)


def fetch_all_protocol_tvls(top: dict[str, list[dict]], refresh: bool = False,
                            max_workers: int = 4):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    seen_slugs: set[str] = set()
    flat: list[tuple[str, str, dict]] = []
    for sector, plist in top.items():
        for p in plist:
            slug = p.get("slug") or p.get("name", "").lower().replace(" ", "-")
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            flat.append((sector, slug, p))

    to_fetch = [(s, sl, p) for (s, sl, p) in flat
                if refresh or not (PROTO / f"{sl}.json").exists()]
    print(f"Fetching {len(to_fetch)} of {len(flat)} protocols "
          f"({len(flat) - len(to_fetch)} cached) ...")

    completed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_protocol, slug, refresh): (sector, slug)
                   for sector, slug, _ in to_fetch}
        for fut in as_completed(futures):
            sector, slug = futures[fut]
            completed += 1
            if completed % 10 == 0 or completed == len(to_fetch):
                rate = completed / max(time.time() - t0, 0.01)
                print(f"  [{completed}/{len(to_fetch)}] {sector:14s} {slug}  "
                      f"({rate:.1f}/s)")
    return flat


# %% Step 5 — build sector × date TVL panel --------------------------------

def build_sector_tvl_panel(flat_protocols, target_chains: set[str]) -> pd.DataFrame:
    """Daily TVL by sector across all in-scope chains. Lending column =
    supplied (idle + borrowed) to match the analysis denominator."""
    daily = pd.date_range("2018-01-01", AS_OF, freq="D")
    panel = pd.DataFrame(index=daily, columns=list(SECTOR_BUCKETS), data=0.0)
    panel.index.name = "date"

    for sector, slug, _summary in flat_protocols:
        fp = PROTO / f"{slug}.json"
        if not fp.exists():
            continue
        try:
            p = json.loads(fp.read_text())
        except json.JSONDecodeError:
            continue
        include_borrowed = (sector == "Lending")
        s = protocol_chain_tvl(p, target_chains, include_borrowed=include_borrowed)
        if s.empty:
            continue
        s = s.reindex(daily).ffill().fillna(0.0)
        panel[sector] = panel[sector].add(s, fill_value=0.0)
    panel["DeFi"] = panel[list(SECTOR_BUCKETS)].sum(axis=1)
    return panel


# %% Step 6 — headline loss-ratio table -------------------------------------

def gross_net_in_window(hacks_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                        sector: str | None = None,
                        in_scope_only: bool = True,
                        exclude_bridges: bool = True,
                        defi_protocol_only: bool = True) -> tuple[float, float]:
    m = (hacks_df["date"] >= start) & (hacks_df["date"] <= end)
    if in_scope_only:
        m &= hacks_df["in_scope_chain"]
    if defi_protocol_only:
        m &= hacks_df["is_defi_protocol"]
    if exclude_bridges:
        m &= ~hacks_df["bridge_hack"]
    if sector is not None:
        m &= (hacks_df["sector"] == sector)
    sub = hacks_df.loc[m]
    return sub["gross"].sum(), sub["net"].sum()


def avg_tvl_window(panel: pd.DataFrame, col: str,
                   start: pd.Timestamp, end: pd.Timestamp) -> float:
    s = panel.loc[start:end, col]
    s = s[s > 0]
    if s.empty:
        return 0.0
    return float(s.mean())


def loss_ratio_table(hacks_df: pd.DataFrame, panel: pd.DataFrame,
                     as_of: dt.date, window_days: int = 365) -> pd.DataFrame:
    end = pd.Timestamp(as_of)
    start = end - pd.Timedelta(days=window_days)
    rows = []
    for sector in list(SECTOR_BUCKETS) + ["DeFi"]:
        if sector == "DeFi":
            gross, net = gross_net_in_window(hacks_df, start, end,
                                             sector=None,
                                             in_scope_only=True,
                                             exclude_bridges=False)
        else:
            gross, net = gross_net_in_window(hacks_df, start, end,
                                             sector=sector,
                                             in_scope_only=True,
                                             exclude_bridges=(sector != "Bridge"))
        tvl = avg_tvl_window(panel, sector, start, end)
        rows.append({
            "sector": sector,
            "gross_loss_usd": gross,
            "net_loss_usd": net,
            "avg_tvl_usd": tvl,
            "gross_bps": (gross / tvl * 1e4) if tvl > 0 else float("nan"),
            "net_bps":   (net / tvl * 1e4) if tvl > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


# %% Main pipeline ---------------------------------------------------------

def main(refresh: bool = False):
    print("[1/5] Loading base data ...")
    hacks, protocols, chains = load_base(refresh=refresh)
    target_chains = chain_universe(chains)
    print(f"      hacks={len(hacks)}  protocols={len(protocols)}  "
          f"target_chains={len(target_chains)} (all DefiLlama-tracked chains)")

    print("[2/5] Building hack dataframe ...")
    hacks_df = hacks_dataframe(hacks, protocols, target_chains)
    (DATA / "raw" / "defillama").mkdir(parents=True, exist_ok=True)
    hacks_df.to_csv(DATA / "raw" / "defillama" / "hacks.csv", index=False)
    print(f"      first hack: {hacks_df['date'].min().date()}  "
          f"last hack: {hacks_df['date'].max().date()}")

    print("[3/5] Selecting top protocols per sector ...")
    top = pick_top_protocols(protocols, target_chains)
    for s, plist in top.items():
        names = [p["name"] for p in plist[:5]]
        print(f"      {s:14s}  n={len(plist)}  top5={names}")

    print("[4/5] Fetching per-protocol TVL histories (cached) ...")
    flat = fetch_all_protocol_tvls(top, refresh=refresh)

    print("[5/5] Building sector TVL panel and headline table ...")
    panel = build_sector_tvl_panel(flat, target_chains)
    panel.to_csv(DATA / "sector_tvl_panel.csv")

    headline = loss_ratio_table(hacks_df, panel, AS_OF, WINDOW_DAYS)
    print("\n=== Trailing 365d loss ratios (all chains, ending {}) ===".format(AS_OF))
    print(headline.to_string(index=False,
                              formatters={
                                  "gross_loss_usd": "${:,.0f}".format,
                                  "net_loss_usd":   "${:,.0f}".format,
                                  "avg_tvl_usd":    "${:,.0f}".format,
                                  "gross_bps":      "{:.2f}".format,
                                  "net_bps":        "{:.2f}".format,
                              }))
    print(f"\nWrote: {DATA/'raw'/'defillama'/'hacks.csv'}")
    print(f"Wrote: {DATA/'sector_tvl_panel.csv'}")
    print("\nNext step: run ingest_consolidated.py to produce the canonical")
    print("event dataset (data/events_consolidated.csv), then risk_analysis.py")
    print("to produce paper figures.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="bypass on-disk caches and re-fetch from DeFiLlama")
    args = ap.parse_args()
    main(refresh=args.refresh)
