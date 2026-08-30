# Pricing the DeFi Tail

Working paper / conference submission: *Pricing the DeFi Tail: Do
Protocols or Depositors Price Operational Risk?*

DeFi protocols market products that look economically similar to bank
deposits — "earn" accounts, lending supply, vaults, stablecoin LP
positions — but place the residual operational risk on the depositor,
not the protocol. This paper assembles the operational-loss dataset
(USD 9.45 B across 1,074 depositor-facing events, 2020–2026, tagged to
Basel Level-1 event types), fits a per-sector loss-distribution
approach (LDA) to build the tail benchmark, and asks whether protocol
capital buffers or depositor risk premia are sized to it. Neither is:
the four largest buffered Lending venues cover ~5% of their modeled
VaR₉₉.₉, and the venue-level supply-yield premium averages +54 bps
against a 64-bps sector pure premium and a per-protocol tail requirement
34–81% of TVL. The full argument, results, and policy discussion are in
[`paper.pdf`](paper.pdf).

## Target venue

CBT 2026 — 10th International Workshop on Cryptocurrencies and Blockchain
Technology, co-located with ESORICS 2026, Rome, 2026-09-17.
Camera-ready deadline: 2026-08-24. LNCS format, 16-page limit including
references. Submitted via EasyChair through the ESORICS 2026 portal.

## Build the PDF

```sh
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

TeX Live 2026 / `pdflatex`; `latexmk -pdf paper.tex` also works.

## Reproduce the numbers

```sh
python3 -m venv .venv
.venv/bin/pip install numpy pandas scipy matplotlib

# Default: camera-ready pipeline. Applies the sector-mistag correction
# (mistags.json) and the depositor-facing R1..R6 filter, refits per
# sector on the 1,074-event working sample, prints Tables 3-8 and the
# §5 robustness paragraph.
.venv/bin/python code.py

# Optional: legacy full-sample figure generation + risk_summary.json.
# Runs on the unfiltered 1,316-event consolidated set.
.venv/bin/python -c "import code; code.regenerate_figures_and_summary()"
```

To rebuild the intermediate CSVs from raw sources:

```sh
.venv/bin/python main.py                     # → data/sector_tvl_panel.csv
.venv/bin/python events_consolidation.py     # → data/events_consolidated.csv
```

Both intermediate CSVs are committed, so `code.py` runs standalone.

## Data sources

| Tag | Source | Coverage |
|---|---|---|
| defillama | DefiLlama `/hacks`, `/protocols`, `/protocol/{slug}` | DeFi-protocol baseline + per-protocol daily TVL |
| rekt | rekt.news editorial leaderboard | USD-ranked entries |
| kismp | kismp123/DeFi-Security-Incident | markdown post-mortems |
| defihacklabs | SunWeb3Sec/DeFiHackLabs | dated entries with Foundry PoCs |
| blocksec | BlockSec Incidents Library | tx hashes, attacker addresses |
| defi_rekt | de.fi/rekt-database | 4,030 records 2011–2026 |
| slowmist | SlowMist Hacked | 2,100 records 2012–2026 |

## Methodology summary

Median-across-sources loss reconciliation; two-pass dedup (name within
±14 d, then date ±7 d + loss ±10%); inferential sector + Basel Level-1
tagging with a 7-agent sector re-audit (`mistags.json`) and a
depositor-facing R1..R6 rule filter; per-sector POT-GPD severity with
plateau-stability threshold and parametric bootstrap; per-sector NB
frequency; compound NB–GPD LDA capped at largest single-protocol
exposure; parametric-bootstrap IQR on VaR₉₉.₉ (Table 6) methodologically
consistent with the parametric ξ̂ CI (Table 4). Full details in
[`paper.pdf`](paper.pdf) §3–§5.
