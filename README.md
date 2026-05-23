# DeFi Exploit Losses — Operational-Risk Treatment

A quantitative-risk analysis of DeFi protocol operational-risk
events 2020–2026, combining **seven** public sources into a single
canonical dataset of **1,495 deduplicated OpRisk events** with
**USD 11.13 B of gross losses**, classified by sub-sector and by the
**Basel III Level-1 event types** (BCBS d457 / OPE25; carried
forward from Basel II Annex 9). The four-category DeFi-adapted
framework of [Chang et al. (2022)](https://www.soa.org/globalassets/assets/files/resources/research-report/2022/decentralized-finance-protocols.pdf)
is supported as a fixed many-to-one collapse (persisted in the
`soa_category` column for reproducibility).

The headline output is the working paper
**[paper.pdf](paper.pdf)** (LaTeX source: [paper.tex](paper.tex)) —
_"Pareto Pools and Omori Aftershocks: Operational-Risk Capital
Requirements for DeFi, Sector by Sector"_ — with a short-form
companion in **[Summary.md](Summary.md)**.

## Pipeline orchestration

The repository ships with a three-stage pipeline. Each stage caches
its output so re-runs are near-instant.

```
                  ┌─────────────────────────────────────────────┐
                  │ 1. RAW SOURCE DATA  (data/raw/)             │
                  │                                             │
                  │ • DefiLlama API caches    main.py           │
                  │ • rekt.news leaderboard   (curl, cached)    │
                  │ • kismp123 GitHub repo    (git clone)       │
                  │ • DeFiHackLabs GitHub     (git clone)       │
                  │ • BlockSec attack-events  (POST API, cached)│
                  │ • de.fi/rekt-database     (paginated JSON)  │
                  │ • SlowMist Hacked tracker (HTML scrape)     │
                  └────────────────────┬────────────────────────┘
                                       │
                  ┌────────────────────▼────────────────────────┐
                  │ 2. CANONICAL INCIDENT DATASET               │
                  │    ingest_consolidated.py                   │
                  │                                             │
                  │  Normalises each source to a common schema, │
                  │  deduplicates by (name, date ± 14d) +       │
                  │  (date ± 7d, loss ± 10 %), and assigns      │
                  │  every record a DeFi sector, a Basel III    │
                  │  Level-1 event type, and a Chang et al.     │
                  │  (2022) risk category (mechanically derived │
                  │  from the Basel tag). Tags come from source │
                  │  data if present, else inferred from name + │
                  │  technique + description.                   │
                  │                                             │
                  │  Output: data/events_consolidated.csv        │
                  └────────────────────┬────────────────────────┘
                                       │
                  ┌────────────────────▼────────────────────────┐
                  │ 3. QUANTITATIVE-RISK ANALYSIS               │
                  │    risk_analysis.py                         │
                  │                                             │
                  │  Filters to 2020–2026 analysis window;      │
                  │  fits EVT severity (Hill, POT-GPD,          │
                  │  Clauset PL); frequency model (Poisson/NB); │
                  │  compound LDA (VaR, ES); Hawkes self-       │
                  │  excitation; F-N exceedance curve with      │
                  │  cross-field benchmarks.                    │
                  │                                             │
                  │  Output: output/risk_summary.json           │
                  │          figures/r1_*.png … r9_*.png        │
                  └────────────────────┬────────────────────────┘
                                       │
                  ┌────────────────────▼────────────────────────┐
                  │ 4. LaTeX PAPER                              │
                  │    paper.tex  →  paper.pdf                  │
                  │                                             │
                  │  References the figures and the SOA / EVT / │
                  │  LDA / Hawkes / F-N benchmarks. Compiles    │
                  │  with `tectonic -X compile paper.tex`       │
                  │  (or any modern LaTeX engine).              │
                  └─────────────────────────────────────────────┘
```

## Quickstart

```bash
# One-time environment setup
python3 -m venv .venv
.venv/bin/pip install numpy pandas scipy matplotlib

# Stage 1: refresh raw source data (idempotent; only fetches what's missing)
.venv/bin/python main.py            # DefiLlama hack ledger + protocol catalog
                                    # rekt + kismp + DeFiHackLabs + BlockSec
                                    # are pre-cached under data/raw/

# Stage 2: produce the canonical incident dataset
.venv/bin/python ingest_consolidated.py
# → data/events_consolidated.csv  (1,495 records, 2020–2026 in scope)

# Stage 3: run the quantitative-risk analysis
.venv/bin/python risk_analysis.py
# → output/risk_summary.json
# → figures/r1_*.png ... figures/r9_*.png

# Compile the paper (requires tectonic or pdflatex)
tectonic -X compile paper.tex
# → paper.pdf
```

## Data sources

| Tag | Source | Coverage | Cache |
| --- | ------ | -------- | ----- |
| **defillama**   | DefiLlama `/hacks` + `/protocols` + `/v2/chains` + `/protocol/{slug}` | 523 records, all chains, with classification + technique + target-type filtering | `data/raw/defillama/` (DefiLlama API JSON) |
| **rekt**        | rekt.news leaderboard | ~295 USD-ranked entries with audit-firm metadata | `data/raw/rekt/leaderboard.html` |
| **kismp**       | kismp123/DeFi-Security-Incident GitHub corpus | ~820 markdown post-mortems with Date / Protocol / Chain / Loss / Root-Cause tables | `data/raw/kismp_repo/` (git clone) |
| **defihacklabs**| SunWeb3Sec/DeFiHackLabs GitHub catalog | ~680 entries 2021–2025 with date + name + root-cause + Foundry PoCs | `data/raw/defihacklabs_repo/` (git clone) |
| **blocksec**    | BlockSec Security Incidents Library (`POST /api/v1/attack/events`) | 282 records 2023-01–2026-05 with project name, USD loss, chain IDs, transaction hashes, attacker addresses, root-cause taxonomy label, X/Twitter media link | `data/raw/blocksec_api/all.json` |
| **defi_rekt**   | de.fi/rekt-database (commercial security platform) | 4,023 records 2011–2026 via paginated REST API; 1,246 records retained after pre-filtering memecoin rugpulls | `data/raw/defi_rekt/page_*.json` |
| **slowmist**    | SlowMist Hacked tracker (`hacked.slowmist.io`) | 2,100 records 2012–2026 via paginated HTML scrape; 1,225 records retained after parseable-USD filter | `data/raw/slowmist/page_*.html` |

## Canonical event schema (`data/events_consolidated.csv`)

Each row is one deduplicated DeFi-protocol operational-risk event.
Columns:

| Column | Description |
| ------ | ----------- |
| `date`            | ISO date of the incident |
| `name`            | Canonical protocol / incident name |
| `loss_usd`        | Reconciled USD gross loss (source precedence: DefiLlama → kismp → DeFiHackLabs → BlockSec → rekt) |
| `recovered_usd`   | USD returned to victims (where known; DefiLlama provides this) |
| `net_usd`         | `loss_usd − recovered_usd` |
| `chain`           | Best-guess chain (e.g. Ethereum, Solana, BSC, Arbitrum) |
| `sector`          | DeFi sector — Lending, DEX, Bridge, Derivatives, Yield, LiquidStaking, **Stablecoin** (combines algorithmic and non-algorithmic stablecoin protocols), RWA, Other |
| `basel2_category` | **Basel III event type** (BCBS d457 / OPE25; carried forward from Basel II Annex 9) — IF (Internal Fraud), EF (External Fraud: code, key, and auxiliary-infra attacks), EPWS (empty in DeFi), CPBP (Clients/Products/Business Practices), DPA (empty in DeFi), BDSF (Business Disruption/System Failures), EDPM (Execution/Delivery/Process Management) |
| `soa_category`    | Chang et al. (2022) operational-risk category — SC-Technical, SC-Economic, Cyber-Operational, Blockchain-Infrastructure (mechanically derived from `basel2_category` via the Basel→Chang mapping in `ingest_consolidated.py:CHANG_FROM_BASEL`) |
| `classification`  | Source-side high-level cause label (DefiLlama where available) |
| `technique`       | Source-side specific attack-pattern label |
| `description`     | Free-text incident description (mostly kismp Root-Cause / BlockSec attack-type text) |
| `sources`         | Comma-separated source-tag list |
| `source_urls`     | Pipe-separated source URLs for each source-tag |
| `n_sources`       | Number of independent sources confirming the incident |

## Methodology highlights

- **Source-precedence rule for loss amount.** When multiple sources
  report different USD figures, we adopt DefiLlama’s value as
  primary. We override to the cross-source median only when at least
  one alternative source disagrees by more than ±50 %.
- **Two-pass deduplication.** Cluster by normalised name within a
  14-day date window; then re-cluster surviving records by date
  ±7 days and loss ±10 %. The second pass catches cross-named
  duplicates such as “Ronin Bridge” / “Ronin Network (Axie Infinity)”
  or “Cetus” / “Cetus Protocol” / “Cetus CLMM”.
- **Inferential sector + dual-taxonomy risk-category tagging.**
  Every record receives a sector and two parallel risk-category
  labels — Basel III Level-1 (IF, EF, EPWS, CPBP, DPA,
  BDSF, EDPM) and the Chang et al. (2022) four-category collapse
  (SC-Technical, SC-Economic, Cyber-Operational,
  Blockchain-Infrastructure). The Chang category is mechanically
  derived from the Basel category via a fixed lookup; the inferential
  rule chain runs textual evidence **before** the DefiLlama-
  classification mapping so a clear operational signature (“oracle
  misconfiguration”, “cbETH deployment error”, “private key
  compromised”) overrides a generic “Protocol Logic” label.
- **Stablecoin sector.** Newly introduced to combine algorithmic
  stablecoins with collateralised stablecoin issuers, basis-trade-
  backed stablecoins, and protocol-specific stablecoin projects
  (Resolv USR, Beanstalk, Cashio, DEUS DEI, Stream Finance, etc.).
- **Non-DeFi filter.** The new sources (de.fi/rekt, SlowMist) cover
  a much broader scope than DeFi protocols, so an extended
  `NON_DEFI_PROTOCOL` regex removes CEX hacks (Bybit, FTX, Mt Gox,
  BtcTurk, Phemex, etc.), Ponzis (BitConnect, OneCoin, PlusToken,
  Bitcoin Sheikh, JPEX, Africrypt, Thodex), individual-user phishing
  records, wallet-software exploits, and one Web2 financial firm
  (Wirecard) wrongly indexed by SlowMist.
- **Analysis window 2020–2026.** Data are collected back to 2011 for
  historical completeness; analysis is filtered to 2020-01-01 onward
  because DeFi as a meaningful ecosystem did not exist before then.
- **TVL denominator.** Sector-level daily TVL series are built from
  the top-50 DefiLlama protocols per sector (>99 % of category TVL).
  Lending TVL is constructed as *supplied* TVL = idle + borrowed so
  the denominator reflects the capital actually using the market.

## Layout

```
defi-exploit-loss-analysis/
├── paper.tex                  # LaTeX source
├── paper.pdf                  # 25-page rendered paper
├── Summary.md                 # short-form companion
├── main.py                    # Stage 1: DefiLlama API ingestion
├── ingest_consolidated.py     # Stage 2: 7-source consolidation + dual-taxonomy tagging
├── risk_analysis.py           # Stage 3: EVT / LDA / Hawkes / F-N for sector + Basel + Chang
├── data/
│   ├── sector_tvl_panel.csv   # daily TVL by sector (Stage 1, only DeFi
│   │                          #   column used in paper as bps denominator)
│   ├── events_consolidated.csv   # canonical 8-source event dataset (Stage 2)
│   └── raw/                   # cached raw source data per source-tag
│       ├── defillama/         # hacks.csv normalised from DefiLlama /hacks
│       ├── rekt/
│       ├── kismp_repo/
│       ├── defihacklabs_repo/
│       ├── blocksec_api/      # full feed JSON from POST blocksec.com/api/v1/attack/events
│       ├── defi_rekt/         # 41 paginated JSON files from api.de.fi/v1/rekt/list
│       └── slowmist/          # 105 paginated HTML pages from hacked.slowmist.io
├── output/
│   └── risk_summary.json      # quant-risk headline numbers (Stage 3, only output)
└── figures/
    ├── r0_events_scatter.png   # events-over-time scatter (§4)
    └── r1_*.png … r8_*.png     # per-sector / pooled quantitative-risk figures
```

## Key findings

- **EVT shape** `ξ̂ ≈ 1.06–1.25` across thresholds for pooled DeFi
  (n=1,495) — statistically indistinguishable from Moscadelli's
  (2004) Basel II operational-risk range and slightly heavier than
  the Eling & Wirfs (2019) cyber-breach range.
- **Hawkes branching ratio** `n̂ ≈ 0.56` with a 22-day excitation
  half-life on the stationary 2021–2026 subset — Omori–Utsu
  aftershock parameters in the range reported for major active
  fault systems. Full-window fit `n̂ ≈ 0.89` (regime-shift confounded).
- **Pooled LDA**: expected annual loss USD 7.01 B (260 bps of TVL)
  vs VaR₉₉ ≈ USD 54 B (2,001 bps) and ES₉₉ ≈ USD 377 B (14,012 bps).
  ES₉₉.₉ approaches USD 2.45 T in unconstrained simulation — the
  ξ ≥ 1 LDA pathology that drove the Basel Committee to abandon
  internal-model operational-risk capital in 2017.
- **Per-sector capital requirements** (bps of trailing-365d
  sector TVL):
  | Sector | n | ξ̂ | TVL | VaR₉₉ | ES₉₉ |
  |---|---|---|---|---|---|
  | Lending | 193 | 0.98 | USD 104 B | 633 bps | **6,153 bps** (≈USD 64 B) |
  | Bridge | 95 | 0.60 | USD 50 B | 1,002 bps | **2,249 bps** (≈USD 11 B) |
  | DEX | 274 | 0.59 | USD 15 B | 852 bps | **1,905 bps** (≈USD 2.9 B) |
  | Yield | 177 | 0.56 | USD 12 B | 876 bps | **1,682 bps** (≈USD 2.1 B) |
  Bridge / DEX / Yield ES₉₉ sit in the 17–22%-of-TVL band, broadly
  comparable to bank Tier-1 OpRisk capital ratios (8–15% of RWA).
  Lending fits ξ̂=0.98 right at the infinite-mean boundary and
  carries the largest absolute capital requirement.

Full paper: [paper.pdf](paper.pdf).
