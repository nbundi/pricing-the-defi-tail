# Pricing the DeFi Tail — Operational-Risk Capital for DeFi Protocols

A loss-distribution-approach (LDA) calibration of per-sector
operational-risk capital requirements for DeFi protocols,
2020–2026. Seven public event-data sources are consolidated into a
single canonical dataset of **1,495 deduplicated DeFi-protocol
operational-risk events** with **USD 11.18 B of gross losses**,
tagged against the **Basel III Level-1 event-type categories**
(BCBS d457 / OPE25; carried forward from Basel II Annex 9). The
LDA Monte Carlo translates per-sector severity and frequency into
operational-risk capital ratios expressed as basis points of
**protocol TVL** (a protocol cannot lose more than its own users
supplied), directly comparable to bank Pillar II OpRisk capital and
applied protocol-by-protocol to assess the adequacy of Aave V3's
Umbrella safety module, MakerDAO's surplus buffer, and Compound's
reserve factor.

The headline output is the working paper
**[paper.pdf](paper.pdf)** (LaTeX source: [paper.tex](paper.tex)) —
*"Pricing the DeFi Tail: A Loss-Distribution Approach to
Operational-Risk Capital"* — with a short-form companion in
**[Summary.md](Summary.md)**.

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
                  │ 2. CANONICAL EVENT DATASET                  │
                  │    ingest_consolidated.py                   │
                  │                                             │
                  │  Normalises each source to a common schema, │
                  │  deduplicates by (name, date ± 14d) +       │
                  │  (date ± 7d, loss ± 10 %), and assigns      │
                  │  every record a DeFi sector and a Basel III │
                  │  Level-1 operational-risk event-type        │
                  │  category. Tags come from source data if    │
                  │  present, else inferred from name +         │
                  │  technique + description.                   │
                  │                                             │
                  │  Output: data/events_consolidated.csv       │
                  └────────────────────┬────────────────────────┘
                                       │
                  ┌────────────────────▼────────────────────────┐
                  │ 3. QUANTITATIVE-RISK ANALYSIS               │
                  │    risk_analysis.py                         │
                  │                                             │
                  │  Per-sector EVT severity (Hill, POT-GPD,    │
                  │  Clauset PL); per-sector annual frequency;  │
                  │  compound Poisson–GPD LDA Monte Carlo per   │
                  │  sector; protocol-level adequacy figures    │
                  │  for Aave V3, MakerDAO, Compound.           │
                  │                                             │
                  │  Output: output/risk_summary.json           │
                  │          figures/r*.png                     │
                  └────────────────────┬────────────────────────┘
                                       │
                  ┌────────────────────▼────────────────────────┐
                  │ 4. LaTeX PAPER                              │
                  │    paper.tex  →  paper.pdf                  │
                  │                                             │
                  │  References the figures, the per-sector     │
                  │  POT-GPD fits, and the per-sector / per-    │
                  │  protocol LDA capital numbers. Compiles     │
                  │  with `tectonic paper.tex` (or any modern   │
                  │  LaTeX engine).                             │
                  └─────────────────────────────────────────────┘
```

## Quickstart

```bash
# One-time environment setup
python3 -m venv .venv
.venv/bin/pip install numpy pandas scipy matplotlib

# Stage 1: refresh raw source data (idempotent; only fetches what's missing)
.venv/bin/python main.py            # DefiLlama hack ledger + protocol catalog
                                    # + per-sector TVL panel
# rekt + kismp + DeFiHackLabs + BlockSec + de.fi + SlowMist
# caches must be present under data/raw/ (re-fetch is per-source)

# Stage 2: produce the canonical event dataset
.venv/bin/python ingest_consolidated.py
# → data/events_consolidated.csv  (1,495 records, 2020–2026 in scope)

# Stage 3: run the quantitative-risk analysis
.venv/bin/python risk_analysis.py
# → output/risk_summary.json
# → figures/r*.png

# Compile the paper (requires tectonic or pdflatex)
tectonic paper.tex
# → paper.pdf
```

## Data sources

| Tag | Source | Coverage |
| --- | ------ | -------- |
| **defillama**   | DefiLlama `/hacks` + `/protocols` + `/v2/chains` + `/protocol/{slug}` | curated DeFi-protocol baseline with classification + technique + target-type filtering; also the per-protocol daily TVL series used as the LDA denominator |
| **rekt**        | rekt.news editorial leaderboard | ~295 USD-ranked entries with audit-firm metadata |
| **kismp**       | kismp123/DeFi-Security-Incident GitHub corpus | ~820 markdown post-mortems with Date / Protocol / Chain / Loss / Root-Cause tables |
| **defihacklabs**| SunWeb3Sec/DeFiHackLabs GitHub catalog | ~680 entries 2021–2025 with date + name + root-cause + Foundry PoCs |
| **blocksec**    | BlockSec Security Incidents Library | 282 records 2023-01–2026-05 with project name, USD loss, chain IDs, tx hashes, attacker addresses, root-cause taxonomy label |
| **defi_rekt**   | de.fi/rekt-database (commercial security platform) | 4,023 records 2011–2026 via paginated REST API; broader scope (pre-filtered + downstream non-DeFi filter) |
| **slowmist**    | SlowMist Hacked tracker (`hacked.slowmist.io`) | 2,100 records 2012–2026 via paginated HTML scrape; broader scope (downstream non-DeFi filter) |

## Canonical event schema (`data/events_consolidated.csv`)

Each row is one deduplicated DeFi-protocol operational-risk event.

| Column | Description |
| ------ | ----------- |
| `date`            | ISO date of the incident |
| `name`            | Canonical protocol / incident name |
| `loss_usd`        | Reconciled USD gross loss (median across reporting sources) |
| `recovered_usd`   | USD returned to victims (where known; DefiLlama provides this) |
| `net_usd`         | `loss_usd − recovered_usd` |
| `chain`           | Best-guess chain (e.g. Ethereum, Solana, BSC, Arbitrum) |
| `sector`          | DeFi sector — Stablecoin, Lending, DEX, Yield, Derivatives (also absorbs Liquid Staking / restaking protocols at the protocol-event level), Bridge, Other |
| `basel2_category` | **Basel III Level-1 operational-risk event-type category** (BCBS d457 / OPE25; carried forward from Basel II Annex 9) — IF (Internal Fraud / rugpulls), EF (External Fraud: smart-contract code attacks, credential compromise, auxiliary-infrastructure), CPBP (Clients/Products/Business Practices: oracle / governance / mechanism abuse), EDPM (Execution/Delivery/Process Management: deployment errors), BDSF (Business Disruption/System Failures: chain-level). EPWS (Employment Practices & Workplace Safety) and DPA (Damage to Physical Assets) are retained in the schema but empty for autonomous DeFi protocols by construction. |
| `soa_category`    | Mechanical many-to-one collapse of `basel2_category` to a coarser four-category label (SC-Technical, SC-Economic, Cyber-Operational, Blockchain-Infrastructure) persisted for downstream consumers; not used in the current paper. |
| `classification`  | Source-side high-level cause label (DefiLlama where available) |
| `technique`       | Source-side specific attack-pattern label |
| `description`     | Free-text incident description |
| `sources`         | Comma-separated source-tag list |
| `source_urls`     | Pipe-separated source URLs for each source-tag |
| `n_sources`       | Number of independent sources confirming the incident |

## Methodology highlights

- **Median-across-sources reconciliation.** When multiple sources
  report the same event with different USD figures, the reconciled
  loss is the *median* of the reported amounts. Robust to
  single-source over- or under-reporting and avoids the implicit
  source-quality ranking that a precedence rule would impose.
- **Two-pass deduplication.** Cluster by normalised name within a
  14-day date window; then re-cluster surviving records by date
  ±7 days and loss ±10 %. The second pass catches cross-named
  duplicates such as "Ronin Bridge" / "Ronin Network (Axie
  Infinity)" or "Cetus" / "Cetus Protocol" / "Cetus CLMM".
- **Inferential sector + Basel-III risk-category tagging.**
  Every record receives a DeFi sub-sector and a Basel III Level-1
  event-type category. Tags come from source data when present and
  are otherwise inferred from name + technique + description against
  an explicit regex rule chain. The inferential chain runs textual
  evidence **before** the DefiLlama-classification mapping so a
  clear operational signature ("oracle misconfiguration", "cbETH
  deployment error", "private key compromised") overrides a generic
  "Protocol Logic" label.
- **Sector taxonomy.** Seven sectors: Stablecoin, Lending, DEX,
  Yield, Derivatives (which absorbs liquid-staking / restaking
  protocols since they share the same collateral-management /
  oracle-dependency OpRisk profile at the event level), Bridge,
  Other. Aligns with DefiLlama's protocol-category convention.
- **Non-DeFi filter.** The broader-scope sources (de.fi/rekt,
  SlowMist) cover beyond DeFi protocols, so a `NON_DEFI_PROTOCOL`
  regex removes CEX hacks (Bybit, FTX, Mt Gox, BtcTurk, Phemex,
  etc.), Ponzis (BitConnect, OneCoin, PlusToken, JPEX, Africrypt,
  Thodex), individual-user phishing records, wallet-software
  exploits, and one Web2 financial firm (Wirecard) wrongly indexed
  by SlowMist.
- **Analysis window 2020–2026.** Data are collected back to 2011
  for completeness; only 8 pre-2020 records survive the
  non-DeFi-protocol filter and they describe proto-DeFi venues whose
  mode of operation differs materially from modern DeFi (centralised
  order-book DEX, single-issuer stablecoin treasury, PoW chain
  economic attacks, pre-AMM token sale). The dataset used for the
  LDA is restricted to 2020-01-01 onward.
- **TVL denominator.** Sector-level daily TVL series are built from
  the top-50 DefiLlama protocols per sector (>99 % of category TVL).
  Lending TVL is constructed as *supplied* TVL = idle + borrowed so
  the denominator reflects the capital actually using the market.
  The LDA simulation runs against sector TVL; the resulting
  $\ES_{99\%}$ ratio is then applied per-protocol because the
  binding upper bound on a single protocol's loss is its own TVL.

## Layout

```
defi-exploit-loss-analysis/
├── paper.tex                  # LaTeX source
├── paper.pdf                  # ~25-page rendered paper
├── Summary.md                 # short-form companion
├── main.py                    # Stage 1: DefiLlama API + TVL panel
├── ingest_consolidated.py     # Stage 2: 7-source consolidation
├── risk_analysis.py           # Stage 3: per-sector EVT + LDA
├── data/                      # (gitignored; rebuild via Stages 1-2)
│   ├── sector_tvl_panel.csv   # daily TVL by sector
│   ├── events_consolidated.csv   # canonical 7-source event dataset
│   └── raw/                   # cached raw source data per source-tag
│       ├── defillama/
│       ├── rekt/
│       ├── kismp_repo/
│       ├── defihacklabs_repo/
│       ├── blocksec_api/
│       ├── defi_rekt/
│       └── slowmist/
├── output/
│   └── risk_summary.json      # quant-risk headline numbers
└── figures/
    ├── r0_events_scatter.png             # events-over-time scatter
    ├── r0b_loss_distribution_by_sector.png
    ├── r0c_rolling_intensity_sector.png
    ├── r0d_rolling_intensity_basel.png
    ├── r1_mean_excess.png … r6b_annual_counts.png   # EVT + frequency diagnostics
    └── r4_gpd_qq_lending.png             # Lending POT-GPD QQ
```

## Key findings

- **Per-sector EVT shape parameters** (POT-GPD, 2020–2026):
  Lending `ξ̂ = 0.93` (just below the infinite-mean boundary,
  statistically indistinguishable from Moscadelli's (2004) Basel II
  banking-OpRisk range); DEX `ξ̂ = 0.62`; Bridge `ξ̂ = 0.60`;
  Yield `ξ̂ = 0.59`; Stablecoin `ξ̂ = 0.30`. Derivatives `ξ̂ = 1.58`
  (`n = 44`, sample-size-limited).

- **Per-sector capital requirements** (compound Poisson–GPD LDA,
  Monte Carlo, expressed as bps of **protocol TVL** — a protocol
  cannot lose more than its own users supplied):

  | Sector | n | ξ̂ | Sector TVL | E[S] bps | VaR₉₉ bps | **ES₉₉ bps** |
  |---|---|---|---|---|---|---|
  | Lending | 192 | 0.93 | USD 104 B | 83 | 562 | **3,828** |
  | Bridge  |  93 | 0.60 | USD 50 B  | 141 | 981 | **2,141** |
  | Yield   | 178 | 0.59 | USD 12 B  | 178 | 907 | **2,082** |
  | DEX     | 274 | 0.62 | USD 15 B  | 180 | 922 | **2,013** |

  Read protocol-by-protocol: ES₉₉ ≈ 19–22% of protocol TVL for
  Bridge / DEX / Yield (broadly comparable to bank Tier-1 OpRisk
  capital ratios of 8–15% of RWA); ES₉₉ ≈ 38% of protocol TVL for
  Lending.

- **Protocol-level adequacy** (modelled ES₉₉ vs publicly-visible
  on-chain reserves, snapshot May 2026):

  | Protocol | Sector | TVL | Modelled ES₉₉ | On-chain buffer | Coverage |
  |---|---|---|---|---|---|
  | Aave V3 | Lending | ~USD 30 B | ~USD 11.5 B | Umbrella stake ~USD 0.3 B | ~3% |
  | MakerDAO | Lending | ~USD 8 B  | ~USD 3.1 B  | Surplus buffer ~USD 0.25 B | ~8% |
  | Compound | Lending | ~USD 3 B  | ~USD 1.1 B  | Per-market reserves ~USD 0.10 B | ~9% |

  None of the three flagship Lending venues clears more than ~10%
  of its modelled 99%-confidence one-year tail event on the
  strength of its on-chain operational-risk reserves alone.

- **Insurance pricing for the long tail.** For the majority of DeFi
  protocols that maintain no formal capital buffer at all, the same
  per-sector LDA output prices a stand-alone protocol
  operational-risk insurance cover: pure premium ≈ 0.8–1.8% of
  protocol TVL per year; risk-loaded one-year premium = the
  per-sector ES₉₉ ratio (19–38% of protocol TVL).

Full paper: [paper.pdf](paper.pdf).
