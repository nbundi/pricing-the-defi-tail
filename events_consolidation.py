"""Consolidate DeFi operational-risk events from seven public sources
and produce a canonical, fully-classified incident master CSV.

Sources (each cached under data/raw/):
    A. DefiLlama        — data/raw/defillama/hacks.csv  (produced by main.py from the
                          public /hacks endpoint; treated as the curated
                          DeFi-Protocol baseline)
    B. rekt.news        — data/raw/rekt/leaderboard.html (HTML scrape of
                          the editorial leaderboard, ~295 ranked exploits)
    C. kismp123 GitHub  — data/raw/kismp_repo/  (per-incident markdown
                          post-mortems, ~820 records)
    D. DeFiHackLabs     — data/raw/defihacklabs_repo/past/{YYYY}/README.md
                          (community per-year incident catalog, ~680
                          records)
    E. BlockSec library — data/raw/blocksec_api/all.json (full feed
                          from POST /api/v1/attack/events on
                          blocksec.com/security-incident, ~280 records
                          with project, USD loss, chain IDs, root-cause
                          label, X/Twitter media link, and tx hashes)
    F. de.fi/rekt-db    — data/raw/defi_rekt/page_*.json (paginated REST
                          API from de.fi/DeFiYield's commercial security
                          platform; ~4 000 records, broad scope including
                          memecoin rugpulls/honeypots that the downstream
                          looks_defi_protocol filter prunes)
    G. SlowMist Hacked  — data/raw/slowmist/page_*.html (Chinese-language
                          security firm SlowMist's public hacked-events
                          tracker; ~2 100 records spanning CEX hacks,
                          DeFi events, Ponzis — non-DeFi noise pruned
                          downstream)

Output:
    data/events_consolidated.csv  — one row per deduplicated event
    with columns:

        date              ISO date
        name              canonical protocol / incident name
        loss_usd          reconciled USD gross loss
        chain             best-guess chain
        sector            DeFi sector — Lending, DEX, Bridge, Derivatives,
                          Yield, LiquidStaking, Stablecoin, RWA, or Other
        soa_category      SOA / Chang et al. (2022) risk category:
                          SC-Technical, SC-Economic, Cyber-Operational,
                          Blockchain-Infrastructure
        technique         attack-pattern label
        description       short free-text description
        sources           comma-separated list of source tags
        source_urls       comma-separated source URLs / git paths
        recovered_usd     USD returned to victims (if known)
        net_usd           loss_usd - recovered_usd

Every record is assigned a sector and a SOA risk category. Records
that do not carry a tag from the source data are tagged by inferring
from the title, technique, and description text against a rule chain
(see CATEGORY_RULES and SECTOR_RULES below).
"""

from __future__ import annotations

import csv
import html
import json
import re
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW  = DATA / "raw"
OUT  = ROOT / "output"
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# Analysis window end. The consolidated dataset is frozen at this date so the
# pipeline is fully reproducible: re-running the loaders picks up newly-scraped
# events beyond the study window, which would silently shift every downstream
# tail fit. Events dated after this cutoff are dropped before the CSV is written.
ANALYSIS_WINDOW_END = date(2026, 5, 29)


# ---------------------------------------------------------------------------
# Sector and SOA category rule sets — applied to records that don't carry
# an explicit tag from the source data. Order matters: the first matching
# rule wins.
# ---------------------------------------------------------------------------

# Mapping DefiLlama category → our sector. We add Stablecoin (combines
# Algo-Stables with non-algorithmic stablecoin protocols).
SECTOR_BUCKETS: dict[str, set[str]] = {
    "Lending":       {"Lending", "CDP", "RWA Lending"},
    "DEX":           {"Dexs", "DEX Aggregator"},
    "Bridge":        {"Bridge", "Cross Chain Bridge", "Chain"},
    "Derivatives":   {"Derivatives", "Liquid Staking", "Liquid Restaking"},
    "Yield":         {"Yield", "Yield Aggregator", "Farm"},
    "Stablecoin":    {"Algo-Stables", "Reserve Currency", "Stablecoin",
                      "Stablecoin Issuer"},
    "RWA":           {"RWA"},
}

# Inferential sector rules: applied to records without a DefiLlama tag.
# Each entry: (regex, sector). The regex is matched against name + technique
# + description text. The first matching rule assigns the sector.
SECTOR_RULES: list[tuple[re.Pattern, str]] = [
    # Hard-named protocol overrides — applied BEFORE the keyword rules so
    # a clear DEX/Lending/etc. protocol can't be mis-classified just
    # because its description mentions "stablecoin" or another bucket
    # keyword. Surfaced from the May 2026 sector audit.
    (re.compile(r"\b(curve\s*finance|curve\s*dex|curve\s*v\d|"
                r"curve(?=\W|$)|vyper|"
                r"uniswap|sushi[- ]?swap|pancake[- ]?swap|"
                r"transit\s*swap)\b", re.I),
     "DEX"),
    # Sky / MakerDAO ecosystem: events affecting the DAI / USDS
    # stablecoin (Vault auctions, DSR, peg break) are Stablecoin sector.
    # Spark Protocol (the lending market) is handled separately in the
    # Lending rule. Match only the precise MakerDAO/Sky names — "DAO
    # Maker" is a different protocol (launchpad) and must not match.
    (re.compile(r"\b(makerdao|maker\s*dao(?!\s*vesting)|"
                r"sky\s*protocol|sky\s*ecosystem|"
                r"usds(?=\W|$)|"
                r"maker\s*vault|sky\s*vault)\b", re.I),
     "Stablecoin"),
    (re.compile(r"\b(saffron\s*finance|snowdog\s*dao|snowdog|"
                r"starstream(?:\s*finance)?|stream\s*finance|"
                r"bearn|dego\s*finance)\b",
     re.I), "Yield"),
    (re.compile(r"\b(ousd|origin\s*usd|origin\s*dollar)\b",
     re.I), "Stablecoin"),
    (re.compile(r"\b(kiloex(?:_perp|\s*perp)?|uni[ -]?btc|"
                r"bedrock(?:\s*restaking)?|btcfi)\b",
     re.I), "Derivatives"),
    (re.compile(r"\b(vee[. ]?finance|echo\s*protocol|"
                r"easy[- ]?fi|easyfi|"
                # Spark Protocol is the Sky / MakerDAO lending product
                # (uses sDAI as collateral). Sector = Lending.
                r"spark(?:\s*protocol|\s*lend)?)\b",
     re.I), "Lending"),
    # Mango Markets stays Derivatives via DefiLlama tag; no override needed.
    # Stablecoin first — names containing "USD", "USR", "FRAX", "DAI",
    # "MIM", "UST", "stable", or known stablecoin issuers
    (re.compile(r"\b(usr|usd[ck0nrt]|usde|frax|dai|mim|ust\b|tribe|fei\b|"
                r"beanstalk|magic\s*internet\s*money|reflexer|"
                r"resolv|sky\s*dollar|deus\s*finance|hund?red\s*dollar|"
                r"hope\s*finance|cashio|stablecoin|stable\s*coin|"
                r"algo[- ]?stables?|"
                r"esd|basis\s*cash|empty\s*set|float\s*protocol|"
                r"feidao|acala|elephant\s*money|"
                r"neutrino|usdn\b|wave\s*flow|"
                r"iron\s*finance|titanium|mim[- ]?spell|kava\s*mint|"
                r"angle\s*protocol|qidao|qi\s*dao|"
                r"ethena|fdusd|tusd|husd|gho\b|crvusd|lusd|"
                r"mountain\s*protocol|usual\s*protocol|usdb\b|"
                r"depeg|de[- ]?peg)\b", re.I), "Stablecoin"),
    # Lending
    (re.compile(r"\b(aave|compound|euler|cream\s*finance|cream\s*lending|"
                r"radiant|venus|sturdy|silo|sonne|moonwell|notional|"
                r"hundred[ -]?finance|warden|granary|teller|onyx|"
                r"benqi|inverse|midas|liquity|maker[ -]?dao|abracadabra|"
                r"agave|geist|tarot|0vix|valas|lend|lending|borrow|"
                r"perpetual\s*loans?|cdp\b|debt[ -]?market|"
                r"bzx|fulcrum|"
                r"morpho|spark[ -]?protocol|fluid|frax[ -]?lend|"
                r"compoundd?[ -]?finance|cre?am[ -]?v\d|"
                r"rari\s*(capital|fuse|pool)|fortress\s*loans?|"
                r"hashflow\s*lending|lend(?:fi|hub|fish))\b", re.I), "Lending"),
    # Bridge
    (re.compile(r"\b(bridge|bridg(e|ing)|cross[ -]?chain|wormhole|"
                r"layerzero|axelar|stargate|nomad|allbridge|multichain|"
                r"polynetwork|poly[ -]?network|portal|orbit|qubit|harmony|"
                r"meter|heco|chainswap|rubic|li\.?fi|li\s*finance|"
                r"hyperbridge|socket|squid|debridge|hop|connext|"
                r"thorchain|maya|across\s*protocol|wanchain|"
                r"binance[ -]?bridge|bnb[ -]?bridge|ronin|ioteh|"
                r"pnetwork|qbridge|evodefi|kelp|"
                r"omnichain|teleport|relay\s*chain|warp[ -]?bridge|"
                r"x[ -]?bridge|inter[ -]?chain)\b", re.I),
     "Bridge"),
    # DEX
    (re.compile(r"\b(uniswap|sushiswap|pancakeswap|curve|balancer|"
                r"thorchain\s*swap|raydium|orca|jupiter|cetus|"
                r"trader\s*joe|quickswap|spookyswap|spiritswap|"
                r"dodo|kyber|kyberswap|0x|paraswap|matcha|cowswap|"
                r"camelot|baseswap|aerodrome|velocore|wombat|"
                r"hashflow|integral|swapper|dex(?:[\s-]?aggregator)?|"
                r"amm\b|liquidity\s*pool|"
                r"uranium|swirl|bancor|loopring|saddle|"
                r"hydradex|fed\s*ml|defichain|"
                r"velodrome|solidly|chronos|fenix|equalizer|"
                r"shibaswap|elk[ -]?finance|tomdex|"
                r"mirror\s*protocol|astroport|terraswap|"
                r"mango\s*markets|mango\s*swap|saber|step\s*swap|"
                r"openswap|hydra\s*swap|maiar|"
                r"thala|kanaloa|biswap|apeswap|levana|"
                r"oolong[ -]?swap|atlasdex)\b", re.I), "DEX"),
    # Derivatives
    (re.compile(r"\b(perp|perpetual|derivative|future|option|"
                r"gmx|gns|gains[ -]?network|dydx|drift|hyperliquid|"
                r"jupiter\s*perps|kwenta|polynomial|"
                r"mango[ -]?markets|opyn|hegic|premia|dopex|"
                r"deri|squeeth|lyra|vela|level|rage[ -]?trade|"
                r"synfutures|injective|symmio|aark|"
                r"zomma|panoptic)\b", re.I),
     "Derivatives"),
    # Yield (extended: index funds, vaults, vesting, launchpads,
    # auto-compounders, yield-farms, plus historic yield protocols)
    (re.compile(r"\b(yearn|harvest|beefy|convex|stake[ -]?dao|"
                r"badger|pickle|ribbon|opyn[ -]?vault|element|"
                r"reaper|granary|gamma|index[ -]?coop|indexed\s*finance|"
                r"yield[ -]?aggreg|farm|vault|auto[ -]?compounder|"
                r"pendle|earnpark|finiko|gro\s*protocol|"
                r"alpha\s*homora|alpha\s*finance|impermax|"
                r"goose\s*finance|yfvalue|yfi\b|yfii|yvalue|"
                r"step\s*finance|compounder|furucombo|vesper|idle|"
                r"vulcan|paid\s*network|hedgey|ichi|cap\s*finance|"
                r"saffron\s*finance|stake\s*hound|stakehound|"
                r"value\s*defi|harvest\s*finance|gro\s*protocol|"
                r"bondly|akropolis|grim\s*finance|popsicle|"
                r"booster|level\s*finance|treasure|"
                r"farmers\s*world|axion|nimbus|"
                r"stake\s*pool|staking\s*pool|"
                r"yfdai|umami|pancakehunny|hunny\s*finance|"
                r"penpie|magpie|stargate\s*pool|pikalend|cluster|"
                r"yield|aggregator|launchpad|launchpool|"
                r"bondly|bondsly|ovix\s*vault|wave\s*pool|"
                r"farming|stake|staked\b|earn\b)\b", re.I), "Yield"),
    # Liquid Staking / Restaking
    (re.compile(r"\b(lido|rocketpool|frax\s*ether|sfrxeth|eigenlayer|"
                r"renzo|kelp\s*dao|swell|ether\s*fi|etherfi|stader|"
                r"liquid\s*(staking|restaking)|restaking|"
                r"rsETH|stETH|cbETH|wstETH|frxETH|sfrxETH|"
                r"ankr(?:\s+staking)?|hashing\s*ad\s*space|"
                r"helio|jito|marinade|m1[- ]?stake|"
                r"validator\s*key|validator\s*operator)\b",
     re.I), "Derivatives"),
    # NB: a separate RWA bucket is folded into "Other" by infer_sector(),
    # but we keep an RWA rule so the textual signal is preserved on the
    # record (the source_sector field gets the RWA hint even if the
    # final reported sector is Other).
    (re.compile(r"\b(rwa\b|real[ -]?world[ -]?asset|tokenized\s*(t-?bill|"
                r"treasury|bond)|maple|goldfinch|centrifuge|credix|"
                r"ondo|backed|matrixdock|swarm|untangled|truflation)\b",
     re.I), "RWA"),

    # ------------------------------------------------------------------
    # Specific protocol names that don't match the keyword patterns above
    # but are DeFi protocols with a clear sector. Surfaced from the
    # ``Other'' audit on the consolidated 2020--2026 dataset.
    # ------------------------------------------------------------------
    (re.compile(r"\b(prisma\s*finance|ionic\s*money|fortress\s*(loans?)?|"
                r"saffron\s*finance|cover\s*protocol|"
                r"punk\s*protocol|fortress\s*protocol|deltaprime|"
                r"ola[ -]?finance|inverse\s*finance)\b",
     re.I), "Lending"),
    (re.compile(r"\b(superfluid|zapper|harvest\s*finance|"
                r"furucombo|arbix\s*finance|cap\s*finance|"
                r"compounder|"
                r"akropolis|grim\s*finance|popsicle\s*finance|"
                r"ichi|wonderland|paid\s*network|nimbus|"
                r"yfvalue|yfv|vesper|idle\s*finance|"
                r"hedgey\s*finance|popsicle|pancakehunny|"
                r"value[ -]?defi|ovix|merlin|inverse[ -]?yield|"
                r"bondly|axion|booster|monoxprotocol|"
                r"jimbos\s*protocol|jimboss\s*protocol)\b",
     re.I), "Yield"),
    (re.compile(r"\b(uranium\s*finance|indexed\s*finance|"
                r"maiar\s*dex|raydium|saber|paraswap|"
                r"mirror\s*protocol|mirror\b|"
                r"baseswap|aerodrome|velodrome|elk\s*finance|"
                r"alphaswap|levana\s*perps|raydium|orca\s*finance|"
                r"bancor|kyber\s*network)\b", re.I), "DEX"),
    (re.compile(r"\b(kiloex|kilo[ -]?ex|finnexus|fin\s*nexus|"
                r"deri\s*protocol|squeeth|hegic|"
                r"opyn|premia|dopex|deri|gns|gmx|drift|"
                r"hyperliquid|symmetric\s*market)\b",
     re.I), "Derivatives"),
    (re.compile(r"\b(angle\s*protocol|qidao|qi\s*dao|mim[ -]?spell|"
                r"acala\s*(swap|network)?|fei\b|rai\b|"
                r"stream\s*finance|nightmare\s*on\s*ftm)\b",
     re.I), "Stablecoin"),
    (re.compile(r"\b(ankr|jito\b|lido|rocket\s*pool|kelp\s*dao|"
                r"renzo|swell|ether\s*fi|stader|m[ -]?stake)\b",
     re.I), "Derivatives"),

    # ------------------------------------------------------------------
    # Generic technique-text fallbacks — these catch records that don't
    # name a known protocol but describe behaviour distinctive of a
    # sector. Applied last because they're the weakest signal.
    # ------------------------------------------------------------------
    (re.compile(r"\b(lending\s*pool|lending\s*market|borrow\s*pool|"
                r"undercollateralised\s*borrow|debt[ -]?market|"
                r"collateral\s*manipulation|liquidation\s*bot)\b",
     re.I), "Lending"),
    (re.compile(r"\b(liquidity\s*pool|amm\s*pool|swap\s*pool|"
                r"trading\s*pair|pool\s*invariant|k[ -]?invariant)\b",
     re.I), "DEX"),
    (re.compile(r"\b(yield\s*pool|staking\s*pool\s*vulnerability|"
                r"auto[ -]?compound|farming\s*pool|vesting\s*contract)\b",
     re.I), "Yield"),
    (re.compile(r"\b(perp(s|etual|etuals)?\s*(exchange|market|exploit)|"
                r"perp[ -]?dex|funding[ -]?rate\s*manipulation|"
                r"price[ -]?feed\s*perp)\b", re.I), "Derivatives"),
]


# Mapping from DefiLlama's classification field → SOA category. The SOA
# Risk-category taxonomy. We tag each record with a Basel II Level-1
# operational-risk event type, as defined in Basel II Annex 9 (BCBS
# 2006) and preserved verbatim in the consolidated Basel III
# framework (OPE25). The seven Level-1 categories are:
#
#   IF    Internal Fraud
#   EF    External Fraud (smart-contract exploits, key compromise,
#         phishing, DNS / frontend hijack, ERC20-approval phishing
#         --- all external-attacker vectors)
#   EPWS  Employment Practices & Workplace Safety (empty by
#         construction in DeFi: traditional banking scope does not
#         transfer)
#   CPBP  Clients, Products & Business Practices
#   DPA   Damage to Physical Assets (empty: DeFi has no physical
#         assets)
#   BDSF  Business Disruption & System Failures
#   EDPM  Execution, Delivery & Process Management
BASEL_CATEGORIES: list[str] = [
    "IF", "EF", "EPWS", "CPBP", "DPA", "BDSF", "EDPM",
]

# DefiLlama's six-level cause classification mapped to Basel L1.
BASEL_FROM_DEFILLAMA: dict[str, str] = {
    "Protocol Logic":           "EF",
    "Smart Contract Language":  "EF",
    "Ecosystem":                "CPBP",
    "Infrastructure":           "BDSF",
    "Rugpull":                  "IF",
    "Solver Exploit":           "CPBP",
}

# Inferential Basel II rules, applied to records without a DefiLlama
# classification. Match against name + technique + description. First
# match wins. Order is significant: rules earlier in the list are more
# specific.
CATEGORY_RULES: list[tuple[re.Pattern, str]] = [
    # === IF (Internal Fraud): unauthorised insider activity ===
    (re.compile(r"\b(rugpull|rug[ -]?pull|exit[ -]?scam|insider|"
                r"backdoor[ -]?owner|drain[ -]?owner|owner[ -]?drain|"
                r"upgrade\s*key|self[ -]?destruct|honeypot|"
                r"project\s*owner\s*internal|"
                # External fund manager / curator default — counts as
                # insider misuse from the protocol-OpRisk perspective
                # (the protocol delegated discretionary control)
                r"external\s*fund\s*manager|"
                r"fund\s*manager\s*(default|fail|disclos)|"
                r"curator\s*default|"
                # Founder / single-signer abuse
                r"founder\s*(fled|disappear|absconded|missing)|"
                r"ceo\s*(detained|arrested|fled|absconded)|"
                r"unilateral\s*access)\b", re.I),
     "IF"),

    # === BDSF (Business Disruption & System Failures):
    # chain-level halts, sequencer outages, consensus issues ===
    (re.compile(r"\b(mev|miner\s*extractable|congestion|sequencer|"
                r"chain\s*halt|consensus|reorg|finality|"
                r"durable\s*nonce|cryptographic\s*vuln|"
                r"signature\s*forgery)\b", re.I),
     "BDSF"),

    # === EDPM (Execution, Delivery & Process Management):
    # team-member configuration / deployment / governance-proposal errors
    # (e.g. Moonwell cbETH oracle priced at USD 1.12 by missing-
    # multiplication misconfiguration; governance proposals shipping
    # broken parameters; oracle mis-deployments) ===
    (re.compile(r"\b(misconfigur(ed|ation)|deploy(ment)?\s*"
                r"(mistake|error)|operator[ -]?error|human\s*error|"
                r"configuration\s*(error|mistake|flaw)|missing\s*"
                r"price\s*sanity|missing\s*sanity\s*check|"
                r"missing\s*validation\s*step|incorrect\s*"
                r"initialization|left\s*unprotected|forgot\s*to\s*"
                r"verify|test\s*environment\s*leaked|"
                r"governance\s*proposal\s*passed|"
                r"on[ -]?chain\s*setup\s*error|"
                # Governance / proposal deployment errors (proposals
                # shipping broken parameters or buggy upgrades)
                r"proposal\s*(caused\s*a?\s*loss|shipping|shipped\s*broken|"
                r"bug|broke|broken|caused\s*an?\s*issue|"
                r"introduced\s*a?\s*(?:bug|issue|vulnerab))|"
                r"buggy\s*proposal|broken\s*proposal|"
                # the specific Moonwell cbETH case + comparable patterns
                r"cbeth\s*(collateral|oracle)|"
                r"oracle\s*(misconfig|misconfigur|mis-?deployed|"
                r"deployment\s*error)|"
                r"wrong\s*price\s*feed|stale\s*oracle\s*config|"
                r"unset\s*price|missing\s*multiplication)\b", re.I),
     "EDPM"),

    # === CPBP (Clients, Products & Business Practices):
    # economic-design failures, governance manipulation, oracle gaming,
    # MEV/sandwich, price-manipulation, depeg. Matched before EF so
    # that flashloan-governance and oracle-manipulation events route
    # to CPBP rather than the catch-all EF below. ===
    (re.compile(r"\b(flash[ -]?loan|flash\s*loans?|oracle\s*"
                r"(manipulation|attack|issue)|price\s*manipulation|"
                r"price[ -]?oracle|twap\s*manipulation|"
                r"spot\s*price|sandwich|just[ -]?in[ -]?time\s*"
                r"liquidity|governance\s*(attack|takeover|"
                r"manipulation|exploit)|donate[ -]?to[ -]?reserves|"
                r"collateral\s*manipulation|stable\s*(de[ -]?peg|"
                r"depeg)|mint\s*reserves|share\s*price\s*inflation|"
                r"first[ -]?depositor|lack\s*of\s*liquidity|"
                # Algorithmic stablecoin mechanism failures
                r"algorithmic\s*stablecoin|"
                # Generic depeg (not just "stable depeg")
                r"depeg|de[ -]peg|de-peg)\b", re.I),
     "CPBP"),

    # === EF (External Fraud) — all external-attacker vectors that
    # are not CPBP economic-design failures: smart-contract code
    # bugs (reentrancy, access-control, math errors, signature-
    # verification bypass), credential compromise (private-key /
    # multisig / hot-wallet / signer phishing / social-engineering),
    # and auxiliary-infrastructure attacks (DNS / Cloudflare /
    # frontend hijack, ERC20-approval phishing, address poisoning).
    # Under Basel III L1 these are all a single EF category. ===
    (re.compile(r"\b(reentrancy|reentrant|access[ -]?control|"
                r"missing\s*access|missing\s*(check|input\s*validation|"
                r"validation)|integer\s*(overflow|underflow)|"
                # Cross-chain message-bridge / OFT exploits
                r"layerzero(?:\s*(?:message|oft))?|forged\s*(?:layerzero|message)|"
                r"oft\s*(?:bridge|exploit|attack)|cross[ -]?chain\s*forgery|"
                r"unchecked\s*shift|rounding\s*(error|inconsistency)|"
                r"math\s*(mistake|error)|precision\s*loss|"
                r"logic\s*(flaw|error|bug)|"
                r"proof\s*verifier|signature\s*exploit|"
                r"transfer\s*logic|deposit\s*function|"
                r"approval\s*(exploit|race)|"
                r"arbitrary\s*call|fake\s*token|spoof|"
                r"empty\s*market|business[ -]?logic|"
                r"router\s*exploit|input\s*validation|"
                r"erc[ -]?4626|skim|sync|burn\s*mechanism|"
                r"composable\s*stable\s*pool|"
                r"contract\s*vulnerability|security\s*vulnerability|"
                r"virtual\s*machine\s*vulnerability|"
                r"smart\s*contract\s*bug|"
                # Credential compromise vectors
                r"private\s*key|multisig\s*(compromise|deployment|"
                r"exploit|bypass|takeover|frontrun)|"
                r"signature\s*compromise|key\s*theft|stolen\s*key|"
                r"hot\s*wallet|compromised\s*(admin|signer|validator|"
                r"deployer|wallet)|admin\s*key|deployer\s*key|"
                r"signer\s*compromise|permission\s*stolen|"
                r"account\s*compromise|key\s*leak|"
                r"validator\s*key\s*compromise|backdoor|"
                r"social\s*engineering|spear[ -]?phish(?:ing)?|"
                r"team\s*phishing|signer\s*phishing|"
                r"phishing\s*attack|phishing\s*compromise|"
                r"sim[ -]?swap|sim[ -]?swapping|"
                r"trojan|malware|supply\s*chain\s*attack|"
                r"telegram\s*was\s*hacked|telegram\s*hijack|"
                r"twitter\s*(hack|hijack|compromise)|"
                # Auxiliary-infrastructure vectors
                r"dns(?:\s*hijack|\s*compromise)?|domain\s*hijack|"
                r"cloudflare|"
                r"frontend\s*(hack|attack|compromise|injection|hijack)|"
                r"malicious\s*frontend|fake\s*(website|frontend)|"
                r"ui\s*(hack|attack|compromise|hijack)|"
                r"malicious\s*permit|permit\s*signature(?:\s*phishing)?|"
                r"approval\s*(phishing|fraud|race)|"
                r"erc[ -]?20\s*approval|"
                r"malicious\s*signature|"
                r"address\s*poisoning|clipboard\s*hijack)\b", re.I),
     "EF"),
]

# Backward-compat: legacy soa_category column (Chang et al. 2022
# DeFi-adapted four-category collapse) is still emitted in the master
# CSV so external consumers of older drafts continue to work, but the
# paper analysis is now performed entirely on Basel L1 directly.
CHANG_FROM_BASEL: dict[str, str] = {
    "IF":                 "Cyber-Operational",
    "EF":                 "SC-Technical",
    "EPWS":               "Cyber-Operational",
    "CPBP":               "SC-Economic",
    "DPA":                "Blockchain-Infrastructure",
    "BDSF":               "Blockchain-Infrastructure",
    "EDPM":               "Cyber-Operational",
}

SOA_FROM_DEFILLAMA: dict[str, str] = {
    k: CHANG_FROM_BASEL[v] for k, v in BASEL_FROM_DEFILLAMA.items()
}


_SECTOR_FOLD = {"RWA": "Other", "Algo-Stables": "Stablecoin",
                "LiquidStaking": "Derivatives",
                "Liquid Staking": "Derivatives",
                "Liquid Restaking": "Derivatives"}


# DefiLlama category -> our sector mapping. Used when SECTOR_RULES
# regex fails to find a sector signature in the event text, so that
# a known DefiLlama-catalogued protocol can still be sector-tagged.
_DEFILLAMA_CAT_TO_SECTOR = {
    "Dexs": "DEX", "DEX Aggregator": "DEX", "Liquidity Manager": "DEX",
    "Lending": "Lending", "NFT Lending": "Lending",
    "Onchain Capital Allocator": "Lending", "Risk Curators": "Lending",
    "SoFi": "Lending", "Uncollateralized Lending": "Lending",
    "Yield": "Yield", "Yield Aggregator": "Yield", "Farm": "Yield",
    "Leveraged Farming": "Yield", "Staking Pool": "Yield",
    "Bridge": "Bridge", "Canonical Bridge": "Bridge",
    "Cross Chain Bridge": "Bridge", "Cross Chain": "Bridge",
    "CDP": "Stablecoin", "Algo-Stables": "Stablecoin",
    "Reserve Currency": "Stablecoin", "Basis Trading": "Stablecoin",
}


def _load_defillama_protocol_lookup() -> dict[str, str]:
    """Build a normalised-name -> sector lookup from DefiLlama's protocol
    catalog. Names are normalised by lowercasing and stripping
    non-alphanumeric characters. Returns an empty dict if the catalog
    file is missing (the rest of the pipeline still works, just falls
    back to SECTOR_RULES regex)."""
    fp = DATA / "raw" / "protocols.json"
    if not fp.exists():
        return {}
    try:
        protos = json.loads(fp.read_text())
    except Exception:
        return {}
    out: dict[str, str] = {}
    for p in protos:
        cat = p.get("category", "")
        sector = _DEFILLAMA_CAT_TO_SECTOR.get(cat)
        if sector is None:
            continue
        nk = re.sub(r"[^a-z0-9]+", "", str(p.get("name", "")).lower())
        if not nk:
            continue
        # Prefer the highest-TVL protocol on collision (the long tail of
        # DefiLlama has many small protocols sharing or near-sharing
        # short names; keep the most-recognised one).
        existing_tvl = _LOOKUP_TVL.get(nk, -1)
        tvl = p.get("tvl") or 0
        if tvl > existing_tvl:
            out[nk] = sector
            _LOOKUP_TVL[nk] = tvl
    return out

_LOOKUP_TVL: dict[str, float] = {}
_DEFILLAMA_NAME_TO_SECTOR = _load_defillama_protocol_lookup()


def _defillama_catalog_match(name: str) -> str | None:
    """Exact-name lookup against the DefiLlama protocol catalog."""
    if not _DEFILLAMA_NAME_TO_SECTOR:
        return None
    nk = re.sub(r"[^a-z0-9]+", "", str(name).lower())
    return _DEFILLAMA_NAME_TO_SECTOR.get(nk)


# High-precision sector suffixes in protocol names: many DeFi protocols
# encode their sector in the trailing token of their name (e.g.
# "Foo Swap" -> DEX, "Bar Lending" -> Lending). Applied as a tertiary
# fallback after SECTOR_RULES and the DefiLlama catalog, with patterns
# anchored at word-boundary or end-of-string to keep false positives
# low. These rules deliberately do NOT catch every plausible variant;
# remaining Other-classified events are genuinely sector-ambiguous from
# name alone.
_SUFFIX_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?:^|[\s_-])(?:swap|dex|amm|exchange|"
                r"liquidity\s*manager|liquidity\s*pool)\s*"
                r"(?:v\d+|finance|protocol|router|pro)?$"
                r"|\w+swap$|\w+[- ]?dex$", re.I), "DEX"),
    (re.compile(r"(?:^|[\s_-])(?:lend|lending|borrow|money\s*market|"
                r"credit\s*market|debt|isolated\s*pool)\s*"
                r"(?:protocol|finance|v\d+|market)?$"
                r"|\w+lend(?:ing)?$", re.I), "Lending"),
    (re.compile(r"(?:^|[\s_-])(?:yield|vault|farm(?:ing)?|harvest|"
                r"optimi[sz]er|aggregator|earn|reward)\s*"
                r"(?:protocol|finance|v\d+|aggreg)?$"
                r"|\w+vault$|\w+farm$|\w+yield$", re.I), "Yield"),
    (re.compile(r"(?:^|[\s_-])(?:usd[a-z]?|stable|stablecoin|peg|cdp|"
                r"reserve\s*currency)\s*"
                r"(?:protocol|finance|v\d+)?$"
                r"|\w*usd[a-z]?$|stable\w*$", re.I), "Stablecoin"),
    (re.compile(r"(?:^|[\s_-])bridge(?:\s*v\d+|\s*protocol)?$"
                r"|\w+bridge$", re.I), "Bridge"),
]


def _suffix_keyword_match(name: str) -> str | None:
    """Match a protocol name against high-precision suffix-anchored
    sector keywords (e.g. names ending in 'Swap', 'Lending', 'Vault').
    Returns the inferred sector or None."""
    nm = str(name or "").strip()
    if not nm:
        return None
    for pat, sec in _SUFFIX_RULES:
        if pat.search(nm):
            return sec
    return None


# Manual sector classifications for the top-100 Other-bucket events,
# applied by exact normalized-name match. These are events the regex
# rules + DefiLlama catalog + name-suffix fallback all leave in Other,
# but where the protocol's economic function is identifiable from its
# documentation. The classifications were assigned via a one-time
# protocol-by-protocol audit (LLM-assisted; manually verified for the
# top events by loss). The aim is to compress the Other catch-all
# bucket without over-extending the named sectors.
_MANUAL_SECTOR_OVERRIDES: dict[str, str] = {
    # DEX (43)
    "ibxtrade": "DEX", "odingodofrunes": "DEX", "aqua": "DEX",
    "surgebnb": "DEX", "voltagefinance": "DEX", "arcadiafi": "DEX",
    "gemholic": "DEX", "sushimisodutchauction": "DEX", "bananagun": "DEX",
    "pcash": "DEX", "feg": "DEX", "fegtoken": "DEX", "swervefinance": "DEX",
    "polkatrain": "DEX", "skywardfinance": "DEX", "blocktowercapital": "DEX",
    "swapx": "DEX", "fegex": "DEX", "sudorare": "DEX",
    "gyrofinancearbitrumliquiditymanagementprotocol": "DEX",
    "tradeonorionorionprotocol": "DEX", "mictoken": "DEX",
    "gelatonetwork": "DEX", "crosswise": "DEX", "rabbyswaprouter": "DEX",
    "synaplogic": "DEX", "d3xai": "DEX", "cofixprotocol": "DEX",
    "oxodexpool": "DEX", "cow": "DEX", "dtrinity": "DEX",
    "transitfinance": "DEX", "lauratoken": "DEX", "hackdaohacktoken": "DEX",
    "alienbase": "DEX", "curveburner": "DEX", "newfi": "DEX",
    "linkdao": "DEX", "ups": "DEX", "coinswop": "DEX",
    "mixedswaprouter": "DEX", "xsdwethpool": "DEX", "firebirdpair": "DEX",
    # Yield (33)
    "makinafi": "Yield", "bitcoinreserveoffering": "Yield",
    "uearnpool": "Yield", "gymnetwork": "Yield", "kannagi": "Yield",
    "mangofarmsol": "Yield", "newgoldprotocol": "Yield",
    "defilabs": "Yield", "levyathan": "Yield",
    "thebigcombogrowthdefi": "Yield", "basketdaoorg": "Yield",
    "zeedfinanceyeedtoken": "Yield", "rodeofinance": "Yield",
    "goldminefinance": "Yield", "rehold": "Yield", "agenticfof": "Yield",
    "libertifylibertivault": "Yield", "numa": "Yield", "amun": "Yield",
    "iearn": "Yield", "earningfram": "Yield", "forcedao": "Yield",
    "pokefarm": "Yield", "snksnktokensnkminter": "Yield",
    "opsec": "Yield", "dbxenxentokenburnbasedstaking": "Yield",
    "barleyfinance": "Yield", "xyearn": "Yield",
    "swappstaking": "Yield", "vista": "Yield", "lpmine": "Yield",
    "usdtstakingcontract28": "Yield", "dappsocial": "Yield",
    # Stablecoin (8)
    "curio": "Stablecoin", "fantasmfinance": "Stablecoin",
    "yamdao": "Stablecoin", "dgld": "Stablecoin", "safedollar": "Stablecoin",
    "baocommunity": "Stablecoin", "bankx": "Stablecoin",
    "usualusd0": "Stablecoin",
    # Bridge (2)
    "iotex": "Bridge", "polyhedra": "Bridge",
    # Derivatives (13)
    "siren": "Derivatives", "grandbase": "Derivatives",
    "usdgambitandtlp": "Derivatives", "auctus": "Derivatives",
    "zerogoki": "Derivatives", "predyfinance": "Derivatives",
    "cozyv2": "Derivatives", "synthetify": "Derivatives",
    "dexodusfinance": "Derivatives", "particletrade": "Derivatives",
    "will": "Derivatives", "willtradingprotocol": "Derivatives",
    "thales": "Derivatives",
    # Lending (13)
    "cerdix": "Lending", "compoundfork": "Lending", "pawnfi": "Lending",
    "asterafi": "Lending", "cyrusfinance": "Lending",
    "avolendfinance": "Lending", "bzxprotocol": "Lending",
    "kashi": "Lending", "ddm": "Lending", "goodcompound": "Lending",
    "ktaf": "Lending", "juiceboxv3": "Lending", "rico": "Lending",
    # round-2 LLM pass (small low-cap residual)
    "ethtrustfund": "Yield", "roar": "Yield", "drlvaultv3": "Yield",
    "maestro": "DEX", "pseudoeth": "DEX",
    # DefiLlama source-tag corrections. DefiLlama's category field labels
    # these protocols "Bridge"/"Other", overriding their true identity; the
    # override (checked before the source tag) restores the correct sector.
    # Lendf.me is dForce's money market (Lending); Wasabi is a perps venue
    # (Derivatives); Stake DAO is a yield/liquid-locker protocol (Yield,
    # matching DefiLlama's own tag on the protocol's other incidents).
    "lendfme": "Lending", "wasabiperps": "Derivatives", "stakedao": "Yield",
}


def _manual_override_match(name: str) -> str | None:
    nk = re.sub(r"[^a-z0-9]+", "", str(name).lower())
    return _MANUAL_SECTOR_OVERRIDES.get(nk)


def infer_sector(name: str, technique: str, description: str,
                 default_sector: str) -> str:
    """Return a sector for the record. If default_sector is set
    (DefiLlama-tagged), it is honored after the standard folds:
    DefiLlama's ``Algo-Stables'' becomes our unified ``Stablecoin'',
    and RWA folds into ``Other'' (the RWA bucket is sparsely populated
    and conceptually overlaps with Stablecoin/Yield once tokenised-
    treasury issuers are split out)."""
    haystack = f"{name} {technique or ''} {description or ''}"
    # Manual override list (audited high-value Other-bucket events).
    # Applied first so it can correct an upstream tag.
    manual = _manual_override_match(name)
    if manual:
        return _SECTOR_FOLD.get(manual, manual)
    if default_sector and default_sector not in ("Other", ""):
        return _SECTOR_FOLD.get(default_sector, default_sector)
    for pat, sec in SECTOR_RULES:
        if pat.search(haystack):
            return _SECTOR_FOLD.get(sec, sec)
    # DefiLlama protocol-catalog fallback: exact-name match against the
    # full DefiLlama protocol list (7{,}500+ entries). Catches small
    # protocols where the regex rules above did not fire because the
    # protocol-name did not have a clear sector-signature in its
    # technique/description text but DefiLlama has it catalogued.
    catalog_sec = _defillama_catalog_match(name)
    if catalog_sec:
        return catalog_sec
    # Suffix-keyword fallback: names ending in "Swap", "Lending",
    # "Vault", etc. typically encode their sector. High-precision rules
    # only — see _SUFFIX_RULES for the patterns.
    suffix_sec = _suffix_keyword_match(name)
    if suffix_sec:
        return suffix_sec
    return "Other"


def infer_basel(name: str, technique: str, description: str,
                classification: str) -> str:
    """Return a Basel II Level-1 event type for the record.

    The textual evidence (technique + description + name) is applied
    FIRST, so a clear operational signature (e.g. ``oracle
    misconfiguration``, ``cbETH collateral exploit''-style phrasing,
    ``deployed without sanity check'') overrides DefiLlama's default
    classification --- which sometimes labels a human-misconfiguration
    incident as ``Protocol Logic'' purely because the on-chain symptom
    looked like a contract bug. DefiLlama's classification is used as
    a fallback when textual evidence is silent."""
    haystack = f"{name} {technique or ''} {description or ''}"
    for pat, cat in CATEGORY_RULES:
        if pat.search(haystack):
            return cat
    if classification and classification in BASEL_FROM_DEFILLAMA:
        return BASEL_FROM_DEFILLAMA[classification]
    # Last-resort default: most DeFi events are external attacks on
    # smart-contract code — Basel External Fraud, Technical sub-type.
    return "EF"


def infer_soa(name: str, technique: str, description: str,
              classification: str) -> str:
    """Return a Chang et al. (2022) SOA category by mechanically
    mapping the inferred Basel II category through CHANG_FROM_BASEL."""
    return CHANG_FROM_BASEL[infer_basel(name, technique, description,
                                        classification)]


# ---------------------------------------------------------------------------
# Curated sector re-audit (see mistags.json). A per-sector review corrects
# events whose inferred sector is wrong and flags events that are not DeFi
# protocols at all (memecoins, gaming/NFT tokens, centralized-venue and
# personal-wallet incidents). Non-DeFi events are marked NOT_DEFI and dropped
# in main() so they never enter the consolidated dataset.
# ---------------------------------------------------------------------------

_ALLOW = re.compile(
    r"\bOHM[- ]?fork\b|Olympus(?:DAO)?[- ]?fork|Olympus[- ]?fork|"
    r"\breserve[- ]?currency\b|"
    r"\bsocial[- ]?token(?:\s+(?:staking|project))?\b|\bUBI\s+token\b|"
    r"social[- ]?creator\s+token|"
    r"\bRWA\b|\bDeFi\s+(?:tool|infra|aggregator|dashboard|hack\s+protection|"
    r"insurance|portfolio)\b|"
    r"\btoken\s+streaming\b|\bkeeper\s+network|\btoken[- ]?vesting|"
    r"\boracle\s+(?:project|protocol)\b|\bDAO\s+tooling\b|Colony\s+Network|"
    r"\balgorithmic-reserve|algorithmic\s+reserve\s+currency",
    re.I,
)
_DENY = re.compile(
    r"\bmemecoin\b|\bmeme[- ]?(?:coin|token)\b|\breflection[- ]?token\b|"
    r"\brebase[- ]?token\b|\brebase\s+memecoin\b|"
    r"\bpyramid\b|\bponzi\b|"
    r"\blaunchpad\b|\bIDO\b|\bpresale\b|"
    r"\bgaming\b|\bGameFi\b|\bP2E\b|\bNFT\b|\bmetaverse\b|"
    r"\bwallet\s+(?:approval|phishing|drain|compromise)\b|\bpersonal\s+wallet\b|"
    r"\bSIM[- ]?swap\b|\btwitter\s+(?:compromise|hijack|scam)\b|"
    r"\boff[- ]?chain\s+corporate\b|"
    r"\bDePIN\b|\bAI\s+(?:token|project|agent|network|data)|"
    r"\bimpostor[- ]?token\b|\bfake\s+(?:bridge|token)\b|"
    r"\bsmall(?:[- ]?cap)?\s+BSC(?:/ETH)?\s+token|"
    r"\bcustodial\s+wallet\b|\bcentrali[sz]ed\s+exchange\s+hot\s*wallet|"
    r"\bfitness\s+app\b|\bmove[- ]?to[- ]?earn\b",
    re.I,
)


def build_reassignment_map() -> dict:
    """(source_sector, iso_date, lower_name) -> corrected_sector."""
    blob = json.loads((ROOT / "mistags.json").read_text())
    out: dict[tuple[str, str, str], str] = {}
    for src, entries in blob["source_sectors"].items():
        for m in entries:
            # Kelp's DAO / LayerZero-bridge exploit belongs in Bridge, not
            # Derivatives: the exploited surface was Kelp's bridge integration,
            # not the LRT product itself.
            if "kelp" in m["name"].lower() and src == "Bridge":
                continue
            target = m["correct_sector"]
            reason = m["reason"].lower()
            is_ohm = any(k in reason for k in (
                "ohm-fork", "ohm fork",
                "olympusdao-fork", "olympusdao fork",
                "olympus-fork", "olympus fork",
            ))
            if target == "Other" and src != "Other":
                if _ALLOW.search(m["reason"]):
                    target = "Stablecoin" if is_ohm else "Other"
                elif _DENY.search(m["reason"]):
                    target = "NOT_DEFI"
                else:
                    target = "NOT_DEFI"
            key = (src, m["date"][:10], m["name"].strip().lower())
            out[key] = target
    return out


def apply_sector_reassignment(h: pd.DataFrame) -> pd.DataFrame:
    """Return `h` with the `sector` column updated per the mistag map."""
    rmap = build_reassignment_map()

    def _new(row):
        key = (row["sector"], str(row["date"])[:10], str(row["name"]).strip().lower())
        return rmap.get(key, row["sector"])

    h = h.copy()
    h["sector"] = h.apply(_new, axis=1)
    return h


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

@dataclass
class Incident:
    date: date
    name: str
    loss_usd: float
    chain: str = ""
    classification: str = ""
    technique: str = ""
    description: str = ""
    source_sector: str = ""
    sources: set[str] = field(default_factory=set)
    source_urls: set[str] = field(default_factory=set)
    defillama_known: bool = False
    is_defi_protocol: bool = False
    recovered_usd: float = 0.0
    raw: dict = field(default_factory=dict)


_SUFFIX_STRIP = re.compile(
    r"\s*[-—]\s*(REKT(\s*\d+)?|Rekt(\s*\d+)?|exploit|hack)\s*$",
    re.I,
)

# Common corporate / protocol suffixes that should be stripped before
# clustering so "Uranium" / "Uranium Finance", "Hedgey" / "Hedgey Finance"
# normalise to the same key. Removed iteratively until no more strip.
_ORG_SUFFIX_STRIP = re.compile(
    r"\s+(finance|protocol|network|labs|dao|inc|"
    r"foundation|swap|money)\s*$",
    re.I,
)
# Version markers in the middle / end of names (V1, V2, V3, v4, 2, 3 …).
_VERSION_STRIP = re.compile(r"\s+v?\d+\s*$", re.I)

# Cross-name alias map applied AFTER stripping/lowercasing. Maps a set of
# known equivalent normalized forms onto a single canonical key so that
# multi-source records using alternative brand names (Portal/Wormhole,
# BSC Token Hub/Binance Bridge, BXH/Boy X Highspeed etc.) cluster together
# in pass 1. Conservative — only enter clear, well-known equivalences.
NAME_ALIAS_MAP: dict[str, str] = {
    # Binance Bridge hack (2022-10-06) — multiple alternate forms
    "bnbbridge":                    "binancebridge",
    "bsctokenhub":                  "binancebridge",
    "bnbchainbinancebscbridgebsctokenhub": "binancebridge",
    "bnbchainbscbridge":            "binancebridge",
    # WOO Network DEX product (Mar 2024)
    "woonetwork":                   "woofi",
    # Smaller renames + cross-name pairs surfaced in final audit
    "refnetwork":                   "refswap",     # Ref.Finance alias
    "ovix":                         "0vix",
    "mmfinancecronos":              "mmfinance",
    "madmeerkatfinance":            "mmfinance",
    "onyxv":                        "onyx",
    "onyxdao":                      "onyx",
    "mbutoken":                     "mobiustoken",
    "mobiusdao":                    "mobiustoken",
    "ribbon":                       "aevo",        # Ribbon renamed to Aevo
    "thirdpartysquidroutermodule":  "thirdpartygnosissafemodule",
    # Round-3 cleanup: post-strip aliases for variants that escape merging
    # because normalize_name strips a suffix that leaves an ambiguous key.
    "madmeerkat":                   "mmfinance",   # after "Finance" strip
    "reffinance":                   "ref",         # "Ref.Finance" (no space → no strip)
    "hackepidemic":                 "origin",      # "Hack Epidemic (Origin Protocol)" after parens
    "abracadabramimspell":          "abracadabraspell",
    # Venus same-day same-USD variants (2026-03-15)
    "venusprotocoliv":              "venus",
    "venuscorepool":                "venus",
    "venusprotocol":                "venus",
    "venustoken":                   "venus",
    # Webaverse — Ahad Shams was the founder/victim
    "ahadshams":                    "webaverse",
    # Socket / Bungee — same exploit (Jan 2024)
    "socket":                       "bungee",
    # YO Protocol variants
    "yoyield":                      "yo",
    # UPS token variants
    "ups":                          "utopiasphere",
    # SwilrLend / SwirlLend — typo variants
    "swirllend":                    "swilrlend",
    # Libertify product brand
    "libertivault":                 "libertify",
    # Dot.Finance / Dot Finance
    "dotfinance":                   "dot",
    # USPD = US Permissionless Dollar
    "uspd":                         "uspermissionlessdollar",
    # Palm USD = Palmswap product
    "palmusd":                      "palmswap",
    # Solend renamed Save (Aug 2024)
    "save":                         "solend",
    # MEV Bot variant pairs
    "mevbot":                       "mevbots",
    # Post-strip short-form canonicalizations (each appears in exactly
    # one cluster — safe to canonicalize)
    "woo":                          "woofi",        # WOO Network (WOOFi) → WOOFi Swap
    "mm":                           "mmfinance",    # MM Finance → MM Finance Cronos
    "cut":                          "cuttoken",     # CUT → CUT token (same-day)
    "caterpillarcoin":              "cuttoken",     # CUT/Caterpillar (same exploit)
    "bh":                           "blackhole",    # BH → Black Hole
    "bhtoken":                      "blackhole",
    "cs":                           "cstoken",      # CS → CS Token
    "sir":                          "sirtrading",   # SIR → SIR Trading
    "leveragesir":                  "sirtrading",
    "tmx":                          "tmxtribe",
    "bgm":                          "bgmtoken",
    "cf":                           "creatfuture",
    "ther0ar":                      "roar",         # The R0AR / Roar
    "tmxtribe":                     "tmxtribe",     # canonical
    "tmxtribetoken":                "tmxtribe",
    # Same protocol variants
    "bcetokenpancakeswapbceusdtpool": "bcetoken",
    "pancakeswapbceusdt":           "bcetoken",
    "mimspell":                     "abracadabraspell",
    "r0ar":                         "roar",
    "cloberdex":                    "clober",
    "cloberliquidityvault":         "clober",
    "curvellamalend":               "curvelend",   # Curve LlamaLend variant
    # Round-3 audit: small remaining aliases
    "hbtoken":                      "hermes",       # HERMES (HB Token) → HB Token
    "hermeshbtoken":                "hermes",
    "civnft":                       "civilization",
    "atmtoken":                     "atm",
    "rosefinance":                  "rosa",         # typo variant
    "rosafinance":                  "rosa",
    "brahtopg":                     "brahma",
    # Inverse Finance sDOLA / Curve LlamaLend (March 2026)
    "sdolallamalendmarket":         "curvelend",
    "inversefinance":               "curvelend",
    "inverse":                      "curvelend",   # post-strip form
    # TEDDY DOGE / DRAC Network (July 2022)
    "teddydoge":                    "drac",
    "dracnetwork":                  "drac",
    # Layer2DAO / QuickSwap rugpull (normalized form keeps the "2")
    "layerdao":                     "quickswap",
    "layer2dao":                    "quickswap",
    # Rose/Rosa Finance — both post-strip forms
    "rose":                         "rosa",
    # Final-pass aliases for residual same-event pairs
    "shatacapital":                 "efvault",       # fund manager / vault
    "metaapes":                     "shell",         # Meta Apes uses SHELL
    "tgc":                          "gpu",           # token rename
    # Wormhole (Portal is the Solana brand name) — 2022-02-02
    "portal":                       "wormhole",
    # Boy X Highspeed / BXH (2021-10-30)
    "bxh":                          "boyxhighspeed",
    # CREAM Finance (multiple alternate names)
    "creamlending":                 "cream",
    "creamfinance":                 "cream",
    # Maiar / Elrond (Maiar was the Elrond-chain native DEX)
    "maiardex":                     "elrond",
    "maiar":                        "elrond",
    # Harmony Horizon Bridge (2022-06-23) — multiple alternate spellings
    "harmonyhorizonbridge":         "harmonybridge",
    "horizonbyharmony":             "harmonybridge",
    "harmonyshorizonbridge":        "harmonybridge",
    # Orbit Bridge / Orbit Chain (2023-12-31)
    "orbitchain":                   "orbitbridge",
    # Fei / Rari merger event (2022-04-30 / 05-01) — multiple variant names
    "feiprotocol":                  "raricapital",
    "feirari":                      "raricapital",
    "feirari2":                     "raricapital",
    "rarifuse":                     "raricapital",
    "raricapitalfei":               "raricapital",
    "raricapitalfeiprotocol":       "raricapital",
    "feiprotocolraricapital":       "raricapital",
    # bZx → renamed Ooki (post-rebrand)
    "ooki":                         "bzx",
    "ookiprotocol":                 "bzx",
    # Curve Vyper-compiler hack (2023-07-30) — multiple ways of naming it
    "vyper":                        "curve",
    "curvevyper":                   "curve",
    "curvedex":                     "curve",
    # GMX (different brand suffix variants)
    "gmxv1perps":                   "gmx",
    "gmxio":                        "gmx",
    # dForce / Lendf.me (Lendf was the dForce lending product brand)
    "lendfme":                      "dforce",
    "dforcelending":                "dforce",
    # MonoX (MonoSwap was the brand)
    "monoswapmonox":                "monox",
    "monoswap":                     "monox",
    # LowCarbCrusader was the MEV-bot validator hack (2023-04-03)
    "lowcarbcrusader":              "mevbots",
    # Matcha used SwapNet aggregator path — same exploit reported under both
    "matcha":                       "swapnet",
    # Abracadabra Spell (multiple variants)
    "abracadabraii":                "abracadabraspell",
    "abracadabraiii":               "abracadabraspell",
    "abracadabramimspellcauldron":  "abracadabraspell",
    # Yearn / YFI token (Feb 2021)
    "yfi":                          "yearn",
    "yearniii":                     "yearn",
    "yearnether":                   "yearn",
    # Prisma Finance (LST + Fi variants)
    "prismalst":                    "prisma",
    "prismafi":                     "prisma",
    # WOO Network / WOOFi DEX product (March 2024)
    "woonetworkwoofi":              "woofi",
    "wooswap":                      "woofi",
    "woofiswap":                    "woofi",
    # Visor Finance renamed to Gamma (Dec 2021)
    "gamma":                        "visor",
    "visorfinance":                 "visor",
    # Origin Protocol (Nov 2020) — parens content stripped already
    "originprotocol":               "origin",
    "hackepidemicoriginprotocol":   "origin",
    # Earning.farm / EFVault (Feb 2023)
    "earningfarm":                  "efvault",
}


def normalize_name(name: str) -> str:
    if not name:
        return ""
    # HTML-decode first so "Fei Protocol &amp; Rari Capital" matches
    # "Fei Protocol & Rari Capital" (some sources leak HTML entities).
    s = html.unescape(name.strip())
    s = _SUFFIX_STRIP.sub("", s)
    s = re.sub(r"\([^)]*\)", "", s)         # drop parenthetical descriptions
    # Iteratively strip version markers + organisation suffixes (a name
    # like "Yearn Finance V2" should reduce to "yearn").
    for _ in range(4):
        new = _VERSION_STRIP.sub("", s)
        new = _ORG_SUFFIX_STRIP.sub("", new)
        if new == s:
            break
        s = new
    s = re.sub(r"[^a-zA-Z0-9]", "", s).lower()
    # Apply alias map so well-known equivalents cluster together.
    return NAME_ALIAS_MAP.get(s, s)


def parse_date_flexible(s: str) -> date | None:
    s = (s or "").strip()
    if not s:
        return None
    fmts = ["%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"]
    for f in fmts:
        try:
            return datetime.strptime(s, f).date()
        except ValueError:
            continue
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None


# ---- A. DefiLlama ---------------------------------------------------------

def load_defillama() -> list[Incident]:
    df = pd.read_csv(RAW / "defillama" / "hacks.csv", parse_dates=["date"])
    out: list[Incident] = []
    for _, r in df.iterrows():
        if not (r.get("gross", 0) > 0):
            continue
        out.append(Incident(
            date=r["date"].date(),
            name=str(r["name"]),
            loss_usd=float(r["gross"]),
            chain=str(r.get("chains") or ""),
            classification=str(r.get("classification") or ""),
            technique=str(r.get("technique") or ""),
            description="",
            source_sector=str(r.get("sector") or ""),
            sources={"defillama"},
            source_urls={"https://api.llama.fi/hacks"},
            defillama_known=True,
            is_defi_protocol=bool(r.get("is_defi_protocol")),
            recovered_usd=float(r.get("recovered") or 0),
        ))
    print(f"  defillama       : {len(out)} records")
    return out


# ---- B. rekt.news ---------------------------------------------------------

def load_rekt() -> list[Incident]:
    fp = RAW / "rekt" / "leaderboard.html"
    if not fp.exists():
        print(f"  rekt            : SKIP — cache missing at {fp}")
        return []
    txt = fp.read_text()
    pat = re.compile(
        r'"title":"([^"]+)"[^{}]{0,500}?"rekt":'
        r'\{"amount":(\d+),"audit":"([^"]*)","date":"([^"]+)"\}'
    )
    records = pat.findall(txt)
    out: list[Incident] = []
    NON_DEFI = re.compile(
        r"^(the one that got away|.*\bcex\b.*|.*-\s*mask off|wintermute|"
        r"sbf|bybit|ftx|dmm|wazirx|bitmart|kucoin|coincheck|"
        r"phemex|indodax|bingx|gala|playdapp|munchables|"
        r"atomic\s*wallet|trust\s*wallet|crypto\.com|liquid\s*global|"
        r"lubian)\b", re.I)
    for title, amount, audit, dstr in records:
        if NON_DEFI.match(title):
            continue
        d = parse_date_flexible(dstr)
        if d is None or d.year < 2014 or d.year > 2030:
            continue
        clean = (title.replace(" - REKT", "").replace(" - Rekt", "")
                       .replace(" - rekt", "").strip())
        out.append(Incident(
            date=d, name=clean,
            loss_usd=float(amount),
            description=f"rekt.news entry; audit={audit or 'none'}",
            sources={"rekt"},
            source_urls={"https://rekt.news/leaderboard"},
            raw={"audit": audit},
        ))
    print(f"  rekt            : {len(out)} records "
          f"(after filtering CEX/non-DeFi)")
    return out


# ---- C. kismp123 ----------------------------------------------------------

PAT_KISMP_DATE  = re.compile(r"\|\s*\*\*Date\*\*\s*\|\s*([\d\-/]+)\s*\|")
PAT_KISMP_PROTO = re.compile(r"\|\s*\*\*Protocol\*\*\s*\|\s*([^|]+?)\s*\|")
PAT_KISMP_CHAIN = re.compile(r"\|\s*\*\*Chain\*\*\s*\|\s*([^|]+?)\s*\|")
PAT_KISMP_LOSS  = re.compile(r"\|\s*\*\*Loss\*\*\s*\|\s*([^|]+?)\s*\|")
PAT_KISMP_RC    = re.compile(r"\|\s*\*\*Root\s*Cause\*\*\s*\|\s*([^|]+?)\s*\|")

PAT_EXPLICIT = re.compile(r"\$\s*([\d,]{5,})(?!\.\d*[mMbBkK])\b")
PAT_MILLION  = re.compile(r"\$\s*([\d,]+\.?\d*)\s*million\b", re.I)
PAT_BILLION  = re.compile(r"\$\s*([\d,]+\.?\d*)\s*billion\b", re.I)
PAT_SUFFIX   = re.compile(r"\$\s*([\d,]+\.?\d*)\s*([mMbBkK])\b")
LOSS_CAP_USD = 5e9     # legitimate DeFi single-incident loss never exceeds this


def parse_loss_text(s: str) -> float | None:
    if not s:
        return None
    s = re.sub(r"\([^)]*\)", "", s.strip())
    for pat, mult in ((PAT_BILLION, 1e9), (PAT_MILLION, 1e6)):
        m = pat.search(s)
        if m:
            v = float(m.group(1).replace(",", "")) * mult
            return v if v < LOSS_CAP_USD else None
    m = PAT_EXPLICIT.search(s)
    if m:
        n = float(m.group(1).replace(",", ""))
        if n > 1e4:
            return n if n < LOSS_CAP_USD else None
    m = PAT_SUFFIX.search(s)
    if m:
        num = float(m.group(1).replace(",", ""))
        v = num * {"k": 1e3, "m": 1e6, "b": 1e9}[m.group(2).lower()]
        return v if v < LOSS_CAP_USD else None
    return None


def load_kismp() -> list[Incident]:
    root = RAW / "kismp_repo"
    if not root.exists():
        print(f"  kismp           : SKIP — repo not cloned at {root}")
        return []
    files = sorted(p for p in root.rglob("*.md")
                   if "vulns" not in p.parts and ".git" not in p.parts)
    out: list[Incident] = []
    skipped = 0
    for f in files:
        try:
            txt = f.read_text(errors="ignore")
        except Exception:
            continue
        # filename pattern: YYYY-MM-DD_Protocol_VulnType[_Chain].md
        stem = f.stem.split("_", 3)
        f_date = parse_date_flexible(stem[0]) if stem else None
        f_proto = stem[1] if len(stem) >= 2 else ""
        f_vuln  = stem[2] if len(stem) >= 3 else ""
        f_chain = stem[3] if len(stem) >= 4 else ""

        m = PAT_KISMP_DATE.search(txt)
        d = parse_date_flexible(m.group(1)) if m else f_date
        m = PAT_KISMP_PROTO.search(txt)
        name = (m.group(1).strip() if m else f_proto).strip()
        m = PAT_KISMP_LOSS.search(txt)
        loss_str = m.group(1) if m else ""
        if re.search(r"\b(nominal|theoretical|hypothetical|simulated)\b",
                     loss_str, re.I):
            skipped += 1; continue
        usd = parse_loss_text(loss_str)
        if not d or not name or usd is None or usd <= 0:
            skipped += 1; continue
        m = PAT_KISMP_CHAIN.search(txt)
        chain = (m.group(1).strip() if m else f_chain).strip()
        m = PAT_KISMP_RC.search(txt)
        rc = (m.group(1).strip() if m else "").strip()
        tech = re.sub(r"(?<!^)(?=[A-Z])", " ", f_vuln).strip()
        out.append(Incident(
            date=d, name=name, loss_usd=usd, chain=chain,
            technique=tech, description=rc,
            sources={"kismp"},
            source_urls={f"https://github.com/kismp123/DeFi-Security-Incident/"
                         f"blob/main/{f.relative_to(root)}"},
        ))
    print(f"  kismp           : {len(out)} records ({skipped} skipped)")
    return out


# ---- D. DeFiHackLabs ------------------------------------------------------

# Each entry in the year README looks like:
#     ### 20241227 Bizness - Reentrancy
#     ### Lost: 15.7k USD
#     ...
#     https://x.com/.../...

PAT_DHL_ENTRY = re.compile(
    r"^### (\d{8})\s+([^\n]+?)\s*-\s*([^\n]+?)\n"          # date, name, root-cause
    r"(?:.*?\n)??### Lost:\s*([^\n]+?)\n"                  # lost line
    , re.MULTILINE | re.DOTALL)


def load_defihacklabs() -> list[Incident]:
    root = RAW / "defihacklabs_repo" / "past"
    if not root.exists():
        print(f"  defihacklabs    : SKIP — repo not cloned at {root}")
        return []
    out: list[Incident] = []
    skipped = 0
    for readme in sorted(root.glob("*/README.md")):
        year = readme.parent.name
        txt = readme.read_text(errors="ignore")
        # Year READMEs have many entries. We use a per-entry split.
        entries = re.split(r"\n(?=### \d{8} )", txt)
        for ent in entries:
            m = re.match(r"### (\d{8})\s+([^\n-]+?)\s*-\s*([^\n]+)", ent)
            if not m:
                continue
            d = parse_date_flexible(m.group(1))
            name = m.group(2).strip()
            cause = m.group(3).strip()
            mloss = re.search(r"### Lost:\s*([^\n]+)", ent)
            loss_str = mloss.group(1) if mloss else ""
            usd = parse_loss_text(loss_str)
            if not d or not name or usd is None or usd <= 0:
                skipped += 1; continue
            out.append(Incident(
                date=d, name=name, loss_usd=usd,
                technique=cause,
                description=f"DeFiHackLabs entry ({cause})",
                sources={"defihacklabs"},
                source_urls={f"https://github.com/SunWeb3Sec/DeFiHackLabs/"
                             f"blob/main/past/{year}/README.md"},
            ))
    print(f"  defihacklabs    : {len(out)} records ({skipped} skipped)")
    return out


# ---- E. BlockSec Security Incidents Library ------------------------------

# BlockSec maintains a paginated incident database at
# /security-incident, backed by POST /api/v1/attack/events. Each record
# is a JSON object with: id, project, projectLogo, loss (USD string),
# chainIds[], transactions[]{txnHash, chainId, attacker, label},
# media (X/Twitter post URL), rootCause (free-text label), date
# (Unix milliseconds), poc, rescued. We cache the single full
# response at data/raw/blocksec_api/all.json.

# Chain-ID → human-readable name. Common EVM and a handful of L1/L2.
_BLOCKSEC_CHAIN_NAMES: dict[int, str] = {
    1: "Ethereum", 10: "Optimism", 56: "BSC", 100: "Gnosis", 137: "Polygon",
    250: "Fantom", 8453: "Base", 42161: "Arbitrum", 43114: "Avalanche",
    59144: "Linea", 5000: "Mantle", 81457: "Blast", 80094: "Berachain",
    1101: "PolygonZkEVM", 1284: "Moonbeam", 1285: "Moonriver",
    1666600000: "Harmony", 25: "Cronos", 324: "ZkSync",
    534352: "Scroll", 7777777: "Zora", 728126428: "Tron",
    146: "Sonic", 143: "Monad", 5: "Sui",
    101: "Solana",   # placeholder for Solana (BlockSec uses 5 sometimes)
}


def load_blocksec() -> list[Incident]:
    fp = RAW / "blocksec_api" / "all.json"
    if not fp.exists():
        print(f"  blocksec        : SKIP — API cache missing at {fp}")
        return []
    try:
        data = json.load(open(fp, encoding="utf-8"))
    except Exception:
        print(f"  blocksec        : SKIP — could not parse {fp}")
        return []
    out: list[Incident] = []
    skipped = 0
    for r in data.get("list", []):
        ts = r.get("date")
        if not ts:
            skipped += 1; continue
        d = datetime.fromtimestamp(ts / 1000).date()
        name = (r.get("project") or "").strip()
        loss = float(r.get("loss") or 0)
        if not name or loss <= 0 or loss >= LOSS_CAP_USD:
            skipped += 1; continue
        rescued = float(r.get("rescued") or 0)
        cause = (r.get("rootCause") or "").strip()
        chain_ids = r.get("chainIds") or []
        chain = ", ".join(_BLOCKSEC_CHAIN_NAMES.get(c, str(c))
                          for c in chain_ids)
        # Transactions: collect tx hashes + attacker addresses for context
        tx_summary = []
        for tx in r.get("transactions", []):
            h = (tx.get("txnHash") or "").strip()
            atk = (tx.get("attacker") or "").strip()
            if h and atk:
                tx_summary.append(f"tx={h[:10]}.. attacker={atk[:10]}..")
            elif h:
                tx_summary.append(f"tx={h[:10]}..")
        desc = (f"BlockSec entry ({cause})"
                + (": " + "; ".join(tx_summary[:3]) if tx_summary else ""))
        urls = set()
        media = (r.get("media") or "").strip()
        if media:
            urls.add(media)
        urls.add(f"https://blocksec.com/security-incident?hash="
                 f"{(r.get('transactions') or [{}])[0].get('txnHash','') or r['id']}")
        out.append(Incident(
            date=d, name=name, loss_usd=loss, chain=chain,
            technique=cause, description=desc[:500],
            sources={"blocksec"},
            source_urls=urls,
            recovered_usd=rescued,
        ))
    print(f"  blocksec        : {len(out)} records ({skipped} skipped)")
    return out


# ---- F. de.fi/rekt-database ----------------------------------------------

# de.fi's "Token" category is a memecoin-scam pool — 2.6k records of
# honeypots/rugpulls that pollute dedup. Drop those at ingest unless
# something elevates them to a real DeFi protocol.
_DEFI_REKT_TOKEN_NOISE = lambda cat, parent: (
    (cat or "").strip() == "Token" and (parent or "").startswith("Exit Scam"))

# Map de.fi categories to our sector taxonomy. Compound categories
# (e.g. "Stablecoin,Borrowing and Lending") match on the first hit.
_DEFI_REKT_SECTOR_RULES = [
    ("bridge",                "Bridge"),
    ("exchange (dex)",        "DEX"),
    ("borrowing and lending", "Lending"),
    ("yield aggregator",      "Yield"),
    ("stablecoin",            "Stablecoin"),
]


def _defi_rekt_sector(name_categories: str) -> str:
    s = (name_categories or "").lower()
    for needle, sector in _DEFI_REKT_SECTOR_RULES:
        if needle in s:
            return sector
    return ""


def load_defi_rekt() -> list[Incident]:
    root = RAW / "defi_rekt"
    if not root.exists():
        print(f"  defi_rekt       : SKIP — pages not fetched at {root}")
        return []
    out: list[Incident] = []
    skipped = 0
    for fp in sorted(root.glob("page_*.json")):
        try:
            data = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        for it in data.get("items", []):
            d = parse_date_flexible(str(it.get("date") or ""))
            name = (it.get("project_name") or "").strip()
            loss = float(it.get("funds_lost") or 0)
            cat  = (it.get("name_categories") or "").strip()
            parent = ((it.get("scam_type") or {}).get("name") or "").strip()
            sub    = ((it.get("scam_type") or {}).get("type") or "").strip()
            if not (d and name) or loss <= 0:
                skipped += 1; continue
            if loss >= LOSS_CAP_USD:
                skipped += 1; continue
            if _DEFI_REKT_TOKEN_NOISE(cat, parent):
                skipped += 1; continue
            recovered = float(it.get("funds_recovered") or 0) + \
                        float(it.get("funds_returned") or 0)
            net = (it.get("network") or {}).get("name") or ""
            # Strip HTML from description, keep the prose
            desc_html = it.get("description") or ""
            desc = re.sub(r"<[^>]+>", " ", desc_html)
            desc = re.sub(r"\s+", " ", desc).strip()[:500]
            technique = f"{parent}/{sub}".strip("/") if (parent or sub) else ""
            out.append(Incident(
                date=d, name=name, loss_usd=loss, chain=net,
                technique=technique,
                description=desc or f"de.fi/rekt entry ({technique or 'n/a'})",
                source_sector=_defi_rekt_sector(cat),
                sources={"defi_rekt"},
                source_urls={f"https://de.fi/rekt-database?id={it.get('id')}"},
                recovered_usd=recovered,
            ))
    print(f"  defi_rekt       : {len(out)} records ({skipped} skipped)")
    return out


# ---- G. SlowMist Hacked --------------------------------------------------

PAT_SLOWMIST_LI = re.compile(r"<li>(.*?)</li>", re.S)
PAT_SLOWMIST_DATE = re.compile(r'<span class="time">([^<]+)</span>')
PAT_SLOWMIST_NAME = re.compile(r"<h3><em>Hacked target:\s*</em>([^<]+)</h3>")
PAT_SLOWMIST_DESC = re.compile(r"<em>Description of the event:\s*</em>(.*?)</p>", re.S)
PAT_SLOWMIST_LOSS = re.compile(r"<em>Amount of loss:\s*</em>\s*([^<]*)")
PAT_SLOWMIST_ATK  = re.compile(r"<em>Attack method:\s*</em>([^<]+)")
PAT_SLOWMIST_REF  = re.compile(r'<a href="([^"]+)"[^>]*>View Reference Sources')


def load_slowmist() -> list[Incident]:
    root = RAW / "slowmist"
    if not root.exists():
        print(f"  slowmist        : SKIP — pages not fetched at {root}")
        return []
    out: list[Incident] = []
    skipped = 0
    for fp in sorted(root.glob("page_*.html")):
        try:
            html = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for li in PAT_SLOWMIST_LI.finditer(html):
            body = li.group(1)
            m_date = PAT_SLOWMIST_DATE.search(body)
            m_name = PAT_SLOWMIST_NAME.search(body)
            if not (m_date and m_name):
                continue
            d = parse_date_flexible(m_date.group(1).strip())
            name = m_name.group(1).strip()
            m_loss = PAT_SLOWMIST_LOSS.search(body)
            loss_str = (m_loss.group(1) if m_loss else "").strip()
            m_n = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", loss_str)
            loss = float(m_n.group(1).replace(",", "")) if m_n else 0.0
            m_atk = PAT_SLOWMIST_ATK.search(body)
            atk = m_atk.group(1).strip() if m_atk else ""
            m_desc = PAT_SLOWMIST_DESC.search(body)
            desc = re.sub(r"<[^>]+>", " ", m_desc.group(1)) if m_desc else ""
            desc = re.sub(r"\s+", " ", desc).strip()[:500]
            m_ref = PAT_SLOWMIST_REF.search(body)
            ref = m_ref.group(1) if m_ref else ""
            if not d or loss <= 0 or loss >= LOSS_CAP_USD:
                skipped += 1; continue
            out.append(Incident(
                date=d, name=name, loss_usd=loss,
                technique=atk,
                description=desc or f"SlowMist entry ({atk or 'n/a'})",
                sources={"slowmist"},
                source_urls={ref or f"https://hacked.slowmist.io/?keyword={name}"},
            ))
    print(f"  slowmist        : {len(out)} records ({skipped} skipped)")
    return out


# ---------------------------------------------------------------------------
# Dedup / merge
# ---------------------------------------------------------------------------

def _share_token(a: str, b: str) -> bool:
    """Two normalized names share a meaningful token (>=4 chars)? Used by
    pass 2 to prevent same-date / near-equal-loss but UNRELATED events
    from merging (e.g. DEUS Finance vs a BAYC scam on the same day with
    coincidentally similar losses)."""
    if not a or not b:
        return False
    # If one is a substring of the other after normalization, that's a
    # share (catches Uranium / Uranium Finance, Hedgey / Hedgey Finance,
    # Venus / Venus Protocol etc.)
    if len(a) >= 4 and a in b:
        return True
    if len(b) >= 4 and b in a:
        return True
    return False


def dedup_merge(records: list[Incident],
                date_tol_days: int = 21) -> list[Incident]:
    """Two-pass dedup: name-cluster + (date,amount,name-similarity)-cluster.

    Also computes cross-source loss-amount disagreement statistics for
    multi-source clusters and prints them to stdout. Disagreement is
    measured as (max - min) / median over the per-source loss amounts
    in each cluster with two or more reporting sources."""
    disagreements: list[float] = []

    def _record_cluster(cluster: list[Incident]) -> Incident:
        merged_r = _merge_cluster(cluster)
        srcs = set().union(*(r.sources for r in cluster))
        if len(srcs) >= 2:
            losses = [r.loss_usd for r in cluster if r.loss_usd > 0]
            if len(losses) >= 2:
                med = float(pd.Series(losses).median())
                if med > 0:
                    disagreements.append((max(losses) - min(losses)) / med)
        return merged_r

    groups: dict[str, list[Incident]] = defaultdict(list)
    for r in records:
        n = normalize_name(r.name)
        if n:
            groups[n].append(r)
    pass1: list[Incident] = []
    for grp in groups.values():
        grp.sort(key=lambda r: r.date)
        cluster: list[Incident] = []
        for r in grp:
            if not cluster:
                cluster = [r]; continue
            if (r.date - cluster[-1].date).days <= date_tol_days:
                cluster.append(r)
            else:
                pass1.append(_record_cluster(cluster)); cluster = [r]
        if cluster:
            pass1.append(_record_cluster(cluster))

    # Pass 2: same-date(±7d) AND amount within 10% AND shared name token.
    # The name-token check is essential: without it the pass over-merges
    # unrelated events that happen to land on the same day with similar
    # losses (a real failure seen in May 2026 ingests, e.g. DEUS Finance
    # merged with a same-day BAYC NFT phishing record).
    pass1.sort(key=lambda r: (r.date, -r.loss_usd))
    merged: list[Incident] = []
    skip: set[int] = set()
    for i, r in enumerate(pass1):
        if i in skip:
            continue
        cluster = [r]
        r_norm = normalize_name(r.name)
        for j in range(i + 1, len(pass1)):
            if j in skip:
                continue
            s = pass1[j]
            if (s.date - r.date).days > 7:
                break
            if abs(s.loss_usd - r.loss_usd) > 0.10 * max(r.loss_usd,
                                                          s.loss_usd):
                continue
            if not _share_token(r_norm, normalize_name(s.name)):
                continue
            cluster.append(s); skip.add(j)
        merged.append(_merge_cluster(cluster) if len(cluster) > 1 else r)

    if disagreements:
        d = pd.Series(disagreements)
        n_zero = int((d == 0).sum())
        print(f"  cross-source loss disagreement on {len(d)} "
              f"multi-source clusters:")
        print(f"    mean        : {d.mean()*100:.1f}%")
        print(f"    median      : {d.median()*100:.1f}%  "
              f"({n_zero}/{len(d)} clusters at 0%)")
        print(f"    p75 / p90   : {d.quantile(0.75)*100:.1f}% / "
              f"{d.quantile(0.90)*100:.1f}%")
        print(f"    n > 50% disagreement : {int((d > 0.50).sum())}")

    return merged


def _merge_cluster(cluster: list[Incident]) -> Incident:
    sources = set().union(*(r.sources for r in cluster))
    source_urls = set().union(*(r.source_urls for r in cluster))
    primary = next((r for r in cluster if "defillama" in r.sources),
                   cluster[0])
    losses = [r.loss_usd for r in cluster if r.loss_usd > 0]
    # Median across sources for the reconciled loss amount. Robust to
    # single-source outliers in either direction and avoids the
    # implicit source-quality ranking that a precedence rule would
    # impose.
    gross = float(pd.Series(losses).median()) if losses else 0.0
    defillama_rec = next((r for r in cluster if "defillama" in r.sources), None)
    if defillama_rec is not None:
        defillama_known = True
        is_defi_protocol = defillama_rec.is_defi_protocol
    else:
        defillama_known = False
        is_defi_protocol = False
    return Incident(
        date=primary.date,
        name=primary.name,
        loss_usd=gross,
        chain=primary.chain or next((r.chain for r in cluster if r.chain), ""),
        classification=primary.classification or next(
            (r.classification for r in cluster if r.classification), ""),
        technique=primary.technique or next(
            (r.technique for r in cluster if r.technique), ""),
        description=" | ".join(r.description for r in cluster if r.description)[:500],
        source_sector=primary.source_sector or next(
            (r.source_sector for r in cluster if r.source_sector), ""),
        sources=sources,
        source_urls=source_urls,
        defillama_known=defillama_known,
        is_defi_protocol=is_defi_protocol,
        recovered_usd=max((r.recovered_usd for r in cluster), default=0.0),
        raw={k: v for r in cluster for k, v in r.raw.items()},
    )


# ---------------------------------------------------------------------------
# Pre-merge date corrections
# ---------------------------------------------------------------------------

DATE_CORRECTIONS: list[dict] = [
    # rekt has Drift Protocol on 4/1/2025 with same USD 285m loss as
    # DefiLlama's Drift Trade on 2026-04-01. Same day-of-year, same
    # amount — rekt typoed the year. Pre-correct so dedup catches it.
    {"source": "rekt", "name_match": "drift protocol",
     "from": "2025-04-01", "to": "2026-04-01"},
]


def apply_date_corrections(records: list[Incident]) -> None:
    for c in DATE_CORRECTIONS:
        fr = parse_date_flexible(c["from"])
        to = parse_date_flexible(c["to"])
        nm = c["name_match"]
        for r in records:
            if (c["source"] in r.sources and r.date == fr
                and nm in r.name.lower()):
                r.date = to


# ---------------------------------------------------------------------------
# Filtering: only DeFi-Protocol-relevant incidents
# ---------------------------------------------------------------------------

# Non-DeFi-Protocol name patterns. Records that match these are dropped
# (unless DefiLlama has explicitly tagged them as DeFi-Protocol). This
# captures CEX hacks, wallet hacks, gaming hacks, etc. that the source
# data doesn't carry an explicit target_type filter for.
NON_DEFI_PROTOCOL = re.compile(
    r"\b("
    # Centralised exchanges + custodians
    r"bybit|ftx|ftx\s*group|dmm|dmm\s*bitcoin|wazirx|bitmart|kucoin|coincheck|"
    r"mt\s*\.?\s*gox|phemex|indodax|bingx|btcturk|nobitex|upbit|bitfinex|"
    r"bithumb|liquid(?:\s*global)?|cryptopia|kraken|coinbase|crypto\.com|"
    r"hotbit|fpg|floating\s*point\s*group|gemini|huobi|htx|youbit|gatecoin|"
    r"cred(?:\s+inc)?|youbit|zaif|cashaa|altsbit|youhodler|aax|jpex|"
    r"coinbit|coindeal|empire\s*market|freeway|coinex(?:\s+exchange)?|"
    r"alphapo|bigone|bilaxy|fixedfloat|3commas|ira\s*financial|"
    r"step\s*hot\s*wallet|lastpass|lastpass\s*users|cypher|"
    r"hashing\s*ad\s*space|hashflare|"
    r"upcx|infini|nfprompt|coinspaid|coins\.ph|okex|eterbase|"
    r"coinsbit|coinrabbit|copay|cryptopia|bibox|"
    r"bitkeep|trustpad|onepiece\s*bridge\s*scam|"
    # CEX/centralised platforms surfaced from the May 2026 audit
    r"lykke|lympo|bitforex|astra\s*nova|symbiogenesis|"
    r"sentinel(?:[ -]?(?:dvpn|cosmos))?|"
    r"whale\s*hunter|"
    # Round-3 audit additions
    r"m2(?:\s*exchange)?|bitbns|hounax|"
    r"bored\s*ape\s*yacht\s*club|bayc|friendsies|"
    r"solfire|my\s*big\s*coin|"
    # Web2 ransomware + small CEX
    r"cwt|2gether|"
    # Round-4 audit additions
    r"adspower|sportsbet(?:\.io)?|eoa\s*token\s*theft|"
    r"fake\s*ansem|tbis|titanium\s*blockchain|"
    r"binance\s*user|eigenlayer\s*investor|"
    # MEV trading bots — not DeFi protocols; the bot operator's loss is
    # not a protocol-level OpRisk event
    r"mev[ _-]?bots?(?:\b|_)|rip\s*mev|mevbot|"
    # Records where the protocol cannot be identified
    r"unknown(?:\s*(?:user|protocol|contract))?|unkonwn|"
    r"unverified[ _-]?contr(?:act|acts)?|"
    r"custom\s*lending\s*pool|"
    # Chain-level / L1 events (not protocol-level OpRisk)
    r"casper\s*network|aeternity|terra\s*2(?:\.0)?|"
    r"solana\b|"
    r"jbs|jbs\s*foods|colonial\s*pipeline|"
    r"holograph|"
    # Centralised lenders / OTC desks
    r"celsius(?:\s*network)?|voyager|genesis|blockfi|nexo|crypto\s*capital|"
    r"alameda(?:\s*research)?|wintermute|jump\s*trading|auros|mgnr|"
    r"fireblocks|stakehound|"
    # Ponzis and exit-scam outfits
    r"bitconnect|wotoken|bitclub|onecoin|plustoken|finiko|mining\s*city|"
    r"thodex|africrypt|hyperverse|libra(?:\s+token)?|terra\s*classic|"
    r"squid\s*game|squid\s*token|forsage|oasis\s*mining|davorcoin|"
    r"bitcoin\s*sheikh|ormeus|empiresx|arbistar|solar\s*techno\s*alliance|"
    r"ackerman\s*ponzi|herencia\s*artifex|ichioka\s*ventures|"
    r"blockchain\s*for\s*dog|saturnbeam|afksystem|flash\.sx|breedtech|"
    r"ponzi|"
    # Wallet apps + browser extensions + individual-user incidents
    r"atomic\s*wallet|trust\s*wallet|electrum|edge\s*wallet|metamask\s*phishing|"
    r"exodus\s*wallet|bo\s*shen|chris\s*larsen|hyperliquid\s*user|"
    r"monkey\s*drainer|coindroplet|address\s*poisoning|"
    # Generic end-user incidents that aren't protocol-level
    r"(?:massive\s+|usdc\s*permit\s*signature\s*)?phishing(?:\s+attack)?|"
    r"social\s*engineering\s*scam|"
    # NFT and gaming (out of operational-risk-of-DeFi-protocol scope)
    r"gala\s*games|playdapp|munchables|farmers\s*world|axie\s*infinity|"
    r"roaring\s*kitty|gamee|gifto|pepe|somesing|akutar(?:\s*nft)?|"
    r"mining\s*capital\s*coin|vulcan\s*forged|"
    # Web2 financial firms wrongly indexed by SlowMist/de.fi
    r"wirecard|cna(?:\s+financial)?|"
    # Cross-chain custody and infra that aren't DeFi protocols
    r"mixin(?:\s*network)?|"
    # L1 chain-level minting bugs (not DeFi-protocol operational risk)
    r"ravencoin|"
    # Other CEX / wallet / generic
    r"nirvana|lubian|deltaprime|sentinel\s*(?:dvpn|protocol|cosmos)|"
    r"compromised\s*owner\s*key|josh\s*jones|gifto|gamee|"
    r"u\.?s\.?\s*government[\s-]*controlled\s*wallet|"
    # Pre-DeFi-era ICO tokens (large headline-loss figures usually refer
    # to market-cap drops, not realised funds extracted)
    r"beauty\s*chain|bec\s*token|smt\s*token)\b", re.I)


# DefiLlama occasionally tags clearly-centralised entities as
# `target_type = "DeFi Protocol"`. We keep an explicit override-exception
# list for these so the DefiLlama tag does not save records that the
# name-blacklist would otherwise drop.
DEFILLAMA_TAG_EXCEPTIONS = {"fixedfloat", "vulcan forged", "wintermute",
                            "terra 2.0", "terra2.0", "terra classic"}

# Pattern checks applied separately from NON_DEFI_PROTOCOL because the
# `\b...\b` word boundaries in that regex don't handle names with `_` /
# hex-address suffixes that some catalogs use (e.g. "MEVBot_0x8c2d",
# "Unverified_667d", "UnverifiedContr_0x431abb").
_OUT_OF_SCOPE_PATTERNS = [
    # MEV trading bots — not DeFi protocols
    re.compile(r"mev\s*bot|mevbot", re.I),
    # Records where the protocol cannot be identified
    re.compile(r"^unknown\b", re.I),
    re.compile(r"^unidentified\b", re.I),
    re.compile(r"unkonwn", re.I),
    re.compile(r"^unverified", re.I),
    re.compile(r"custom\s*lending\s*pool", re.I),
    # Memecoin / scam-token pattern: name ends with a (TICKER) suffix
    # where TICKER is 2-6 uppercase alphanumerics (e.g. "SolDragon (DRAGON)",
    # "Fake Safe Token (SAFE)", "ApeCoin (APE)"). Loss is borne by token
    # holders, not by a protocol's user funds; out of scope for
    # protocol-level OpRisk.
    re.compile(r"\(\s*[A-Z][A-Z0-9]{1,5}\s*\)\s*$"),
    # Round-5 audit: confirmed non-DeFi events that slipped through
    # earlier filter (chain-level events, mining pools, Bitcoin ATMs,
    # gambling platforms, wallet-software incidents).
    re.compile(r"^MEV\s*Boost\s*Exploit\b", re.I),
    re.compile(r"^Bitcoin\s*Depot\b", re.I),
    re.compile(r"^BTC\.com\b", re.I),
    re.compile(r"^IOTA\s*wallet\b", re.I),
    re.compile(r"^General\s*Bytes\b", re.I),
    re.compile(r"^Webaverse\b", re.I),
    re.compile(r"^CoinPoker\b", re.I),
    re.compile(r"^ETC\b\s*$"),                  # Ethereum Classic 51% attack
    re.compile(r"^Bitcoin\s*Gold\b", re.I),     # BTG 51% attack
    # NFT-collection mint scams (NFT mints aren't DeFi-protocol OpRisk):
    re.compile(r"^Mutant\s*Ape\s*Planet\b", re.I),
    re.compile(r"^Evolved\s*Apes?\b", re.I),
    re.compile(r"^Big\s*Daddy\s*Ape\s*Club\b", re.I),
    re.compile(r"^Bored\s*Ape\s*Europe\s*Club\b", re.I),
    re.compile(r"^Gutter\s*Cat\s*Gang\b", re.I),
    re.compile(r"^Monkey\s*Kingdom\b", re.I),
    re.compile(r"^Frosties\b", re.I),
    re.compile(r"^HeroCat\b", re.I),
    re.compile(r"^CryptoBike\b", re.I),
    re.compile(r"^Doodled?\s*Dragons?\b", re.I),
    re.compile(r"^Doodle\s*Monkey\b", re.I),
    re.compile(r"^Meta\s*Apes\b", re.I),
    re.compile(r"^Fury\s*of\s*the\s*Fur\b", re.I),
    re.compile(r"^Apache\s*SalesRoom\b", re.I),
    re.compile(r".*NFT\b", re.I),  # Names ending or containing "NFT" tag
    # Pure memecoin / fake-token scams (no protocol mechanics):
    re.compile(r"^Fake\s+\w", re.I),            # "Fake Linea token", "Fake Notcoin", etc.
    re.compile(r"^Day\s*of\s*Defeat\b", re.I),
    re.compile(r"^Lucky\s*[Ss]tar(\s+Currency)?\b", re.I),
    re.compile(r"^GMETA\b", re.I),
    re.compile(r"^SHARPEI\b", re.I),
    re.compile(r"^Safereum\b", re.I),
    re.compile(r"^Encryption\s*AI\b", re.I),
    re.compile(r"^Banksy\b", re.I),
    re.compile(r"^BabyElon\b", re.I),
    re.compile(r"^ElonMVP\b", re.I),
    re.compile(r"^Mango\s*INU\b", re.I),
    re.compile(r"^PEPEP\b", re.I),
    re.compile(r"^Iconics\b", re.I),
    re.compile(r"^SkyVerse\b", re.I),
    re.compile(r"^The\s*Micro\s*Elements\b", re.I),
    re.compile(r"^REALSWAK\b", re.I),
    re.compile(r"^WanderVerse\b", re.I),
    re.compile(r"^Builders\s*NFT", re.I),
    re.compile(r"^Astronaut\b", re.I),
    re.compile(r"^Miss\s*Universe\b", re.I),
    re.compile(r"^land\s*$", re.I),
]


# Round-3 expansion: description-keyed filters that catch the long tail of
# NFT, memecoin, Ponzi, fake-token, presale-rug, KOL-phishing, gambling,
# GameFi/SocialFi, CEX/infrastructure events that slipped past name-only
# filters. Applied against (name + " " + description) per record.
_OUT_OF_SCOPE_DESC_PATTERNS = [
    # NFT collections, marketplaces, and mint scams
    re.compile(
        r"\bnft\b|opensea|cyber\s*kongz|azuki|remilia|milady"
        r"|mutant\s*ape|bored\s*ape|gutter\s*cat|rare\s*bears?"
        r"|meta\s*apes?|doodle|crypto\s*batz|meebits|wilder\s*world"
        r"|jaypegs?|pixel\s*penguin|monkey\s*kingdom|fractal\s*defi"
        r"|\bfloor\s*(protocol|finance)\b|treasure\s*dao|gooniez|quixotic"
        r"|flippaz|super\s*rare|step\s*hero\s*nfts?|nf\s*dao|nft\s*flow"
        r"|cryptobike|big\s*daddy\s*ape|evolved\s*ape|nft\s*marketplace"
        r"|nft\s*mint|p2e|play.?to.?earn|cryptopunks?|\bsudo\s*rare\b"
        r"|meta\s*pets?|bnb\s*heroes?|bubbleworld|automated\s*assassins"
        r"|sebastien\s*tho|trippy\s*?world",
        re.I,
    ),
    # Pure memecoins / shitcoins without identifiable protocol mechanics
    re.compile(
        r"\b(meme|memecoin)s?\b|pump\.?fun|^floki|^baby|pepe|^elon"
        r"|^dogioh|shib(a|adao|aitoken)?|\bsmurf|\binu\b|\bshiba\b"
        r"|\bbonk|condomsol|superfortune|babydoge",
        re.I,
    ),
    # Gambling / casino / lottery
    re.compile(
        r"casino|gambling|roulette|jpulse|fortune\s*wheel|ether\s*crash"
        r"|metawin|gmbl\.computer|\bdice\b|\blottery\b|hype\s*bet",
        re.I,
    ),
    # GameFi / metaverse / play-to-earn
    re.compile(
        r"gamefi|\bgmee\b|\bgala\b|^xai\b|mmorpg|ember\s*sword"
        r"|topgoal|karastar|metaverse|pirate.*pirate|farcana"
        r"|cashverse|pokemon-?fi|\bp2e\b|brand\s*new\s*quest|webaverse"
        r"|monkey\s*kindom|arenaplay|doglands",
        re.I,
    ),
    # SocialFi
    re.compile(
        r"friend\.?tech|stars?\s*arena|twit\s*fi|audius|galxe"
        r"|polycule|debox|social-?fi|seascape\s*network",
        re.I,
    ),
    # Wallet / individual / KOL / phishing victims
    re.compile(
        r"wallet\s*hack|wallet\s*compromise|phishing|address\s*poison"
        r"|drainer|pig\s*butcher|romance\s*scam|kol\s*wallet"
        r"|individual.*phishing|^vitalik|^kevin\s*rose|luke\s*dashjr"
        r"|^mark\s*cuban|suji\s*yan|bill\s*murr?ay|nikhil\s*gopalani"
        r"|steven\s*galanis|\bdeekay\b|\bjrny\b|jon\s*prosser"
        r"|ivan\s*bianco|\bbitlord\b|crypto\s*?rom|gana\s*payment"
        r"|fetch\.?ai\s*tokens?\s*from\s*phishing|hideyoapes"
        r"|cryptobatz|raresetters",
        re.I,
    ),
    # CEX / centralised wallets / infrastructure / oracle / L1
    re.compile(
        r"ledger\s*(connect|live)?|slope\s*finance|starkware"
        r"|fantom\s*foundation|tornado\s*cash|chainlink\b"
        r"|low\s*carb\s*crusader|mev[- ]?boost|patricia|^inx\b"
        r"|\baltilly\b|^mina\s*protocol\b|^circle\b|\brobinhood\b"
        r"|\bcoinmarketcap\b|huge\s*gas\s*fee|\bunibot\b|^bitbot"
        r"|solareum|btcm\s*app|profanity|\bvanity\s*address\b"
        r"|^twitter$|\bmicrostrategy\b|new\s*market\s*trading"
        r"|telcoin|bitcoin\s*mission|cloud\s*ai|android\s*application"
        r"|launchzone|bitbrowser|fetch\.ai\s*has\s*been\s*granted"
        r"|altilly|^bitlord|debot",
        re.I,
    ),
    # Ponzi / pyramid / investment-fraud schemes
    re.compile(
        r"\bponzi\b|pyramid\s*scheme|investment\s*scam"
        r"|fraudulent\s*platform|multi.?level\s*marketing"
        r"|detrade|vitae|\biearn\s*bot\b|cryptorom",
        re.I,
    ),
    # Private-key / mnemonic leaks of token deployers (token-only rugs,
    # not protocol OpRisk)
    re.compile(
        r"private\s*key\s*(leak|compromise|theft|stolen|leakage)"
        r"|compromised\s*(private\s*key|mnemonic|wallet)"
        r"|mnemonic\s*(phrase\s*)?leak|deployer\s*key\s*(was\s*)?compromised"
        r"|deployer\s*wallet\s*(was\s*)?attacked"
        r"|metamask\s*wallet\s*.*\s*compromised"
        r"|multisig\s+wallet\s+.*compromised"
        r"|hot\s*wallet\s*compromis",
        re.I,
    ),
    # Fake-token / impersonation scams
    re.compile(
        r"fake\s+\w+\s*token|fake\s+\w+\s+governance|phishing\s+token"
        r"|imitat(ing|ed)|impersonat|fake\s+migration"
        r"|fake\s+(airdrop|announcement)",
        re.I,
    ),
    # Presale / launchpad / IDO rug-pulls and exit scams
    re.compile(
        r"\bpresale\b\s*(rug|scam|exit)|exit\s*scam"
        r"|rug.?pull(ed|ing)?|rugpull|\brug(\b|ged)|soft\s*rug"
        r"|developer\s*stole|deployer\s*dump|deployer.*rug"
        r"|devs?\s*disappeared|price.*dropped.*by\s*more\s*than\s*9\d"
        r"|dropped.*9[0-9]\s*\.?\s*\d*\s*%"
        r"|fundraising\s*platform|launchpad\s+(rug|scam|exit)"
        r"|token\s*sale.*scam|launchpad.*compromis"
        r"|dao\s*maker\s*vesting|raccoon\s*network|freedom\s*protocol"
        r"|ido\s*platform|the\s*bribe\s*protocol|jumpnfinance"
        r"|union\s*capital|new\s*market\s*trading|^nexera$"
        r"|cryptobottle|drained\s*by\s*design|money\s*for\s*nothing"
        r"|decentraworld|forest\s*tiger\s*pro|^hege\s*coin|^hege$"
        r"|pok[eé]monfi|^vpanda|^goat$|^multi\s*financial$|^xpet$"
        r"|metaland\s*dao|filesystemvideo|galaxyfoxtoken"
        r"|^ticker$|crypto\s*burgers|^fff$|^ipc(\s*token)?$|^dcf(\s*token)?$"
        r"|^vow\s*\(vowcurrency\)?|^bond\s*protocol$"
        r"|^sss\s*\(blast\)|condom\s*sol",
        re.I,
    ),
    # Generic placeholders / unidentified rekt.news entries
    re.compile(
        r"^nan$|an\s*un.?sol.?ved\s*mystery|drained\s*by\s*design"
        r"|money\s*for\s*nothing|^geniusai|loot\s*dao"
        r"|^pundi\s*ai|^foom\s*cash|^mobius\s*token|^olaxbt"
        r"|^wilder$|zero\s*name\s*service|^zenon\s*network|^lndfi"
        r"|^airdao|^cf\s*token|^dao_officials|^kper\s*network"
        r"|^normie$|^miner$|^inferno$|^unlock\s*protocol"
        r"|^tsuru$|^vestradao|^svt\s*\(solvent|^ycdeal3|^duo$"
        r"|^p719\s*token|^mev_|^mtdao|^bra\s*token"
        r"|^market$|^rant$|^dn404|^stoic_dao|^distx$"
        r"|^thunder$|^magix$|^nblgame|nebula\s*revelation"
        r"|^shidao$|^starman$|^maid$|^laxo\s*token"
        r"|^doodi\s*pals|^ratz\s*club|^twit\s*fi|^z123$"
        r"|^closedai|^bubai$|^yoda$|^daedalus\s*dao"
        r"|pandorachaindao|^zora\s*token|^qnt\s*\(reserve|^ara\s*blocks"
        r"|^atk$|^utopia$|^mad$|^yon$|^bnbpay$|^autochain\s*global"
        r"|^ffist$|^osn(\s*token)?$|^modelpitoken"
        r"|^ddc(\s*token)?$|^dpc$|^ipo$|^wetc$|^prcl$"
        r"|^drlvaultv3$|^mars$|^ontr$|^aphrodite\s*protocol"
        r"|^subquery\s*network|^prxvt$|^trustthetrident"
        r"|^selltoken01|citydao|citadel\.one|elontroll"
        r"|^melo$|^dman$|^nfdao$|^rice$|^btc24h$"
        r"|xblast|^cbdao$|^babyfido|^wgpt\s*token"
        r"|^oraclebnb$|^stimmy$|^pirate\s*x\s*pirate"
        r"|^kub\s*split|friendchipstech|^ranger$|^hgm"
        r"|neverfall\s*protocol|^omni\s*real\s*estate"
        r"|karastar|^azukidao|mainnetsettler|^labubu$"
        r"|hackerdao|dogioh|^ast$|^oly$|^saga\s*dao$"
        r"|^fantom$|^ais$|^mintrisesprices|victor\s*the\s*fortune"
        r"|^hydt$|^the\s*honest\s*venture|^ght$|^pepg$"
        r"|^beast$|^t-mobile$|^nody$|^eden\s*network"
        r"|^carolprotocol|^xsij|^life\s*protocol|^vtf\s*token"
        r"|^inuko$|^game$|^aquadao|^magic$|^cheebs"
        r"|^tableland|^coredefinance|^genomesdao|^iprotocol"
        r"|^acb$|^zoomercoin|^bfctoken|^liveartx|^excommunity"
        r"|^lunafi$|^haribo$|^holy\s*trinity|jokintheboxeth"
        r"|^minterpro|^airwa$|^rugged\s*art|^apemaga"
        r"|^gpu$|^3913token|^carrot\s*token|^shibatoken"
        r"|^btc20token|^bonkfun|^bitpaidio|^bitcoin\s*atm"
        r"|^omnipus$|^unidark|^cat\s*nation|^fpr\s*token"
        r"|^yzer$|^yystoken|^98token|^web3memes|^un\s*token"
        r"|^last\s*kilometer|^horizon\s*finance|^gss$|^verify$"
        r"|^pltd$|^eee$|^chivo$|jpulsepot|^bnbs"
        r"|^hackathon$|contract_0x|^thenftv2|^mchaincapital"
        r"|^blackgold|^sbr\s*token|^wall\s*street\s*memes|^weco$|^tch$"
        r"|^bscgem|^eip-7702|^bitcoin\.org|^next\s*earth"
        r"|^chisale|^wifcoin|^dragonball|standing\s*on\s*bizness"
        r"|^introspection\s*token|^coindaq|^gfa\s*token|^ehive"
        r"|^kr$|^crb2|^winr$|^erc20transfer|^zs$|^gfa$"
        r"|^minestm|^ncd$|^it\s*token|^vds$|^x319$"
        r"|^micdao|^redkeyscoin|^red\s*keys\s*game|^mirage\s*finance"
        r"|^fractal\s*defi|redkeysgame|^gmgn$|^hors$"
        r"|^supermariobros|^lootbot|^duckvader|mintofinance"
        r"|^vrug$|^saturn$|^sut$|^buidl$|^crew$|^vinu$"
        r"|veil\s*cash|^arcade$|^btb\s*token"
        r"|^mamo$|^medamon|^decrypt$|^slurpycoin|^burntbubba"
        r"|^uerii\s*token|^kest$|^ethfin$|^gaia\s*protocol",
        re.I,
    ),
]


def looks_defi_protocol(r: Incident) -> bool:
    name = (r.name or "")
    name_lc = name.lower().strip()
    # DefiLlama's structured DeFi-vs-not classifier is the most
    # authoritative single signal, but we override it for the
    # exception list (entities mis-classified upstream).
    if r.defillama_known and name_lc not in DEFILLAMA_TAG_EXCEPTIONS:
        return r.is_defi_protocol
    # All other sources (including de.fi/rekt's name_categories tag): the
    # name-regex blacklist always wins. de.fi/rekt's sector taxonomy
    # includes centralised lenders / exchanges, so a non-empty
    # source_sector is NOT a reliable DeFi-vs-not signal.
    if NON_DEFI_PROTOCOL.search(name):
        return False
    # MEV / Unknown / memecoin out-of-scope patterns (separate from the
    # main blacklist to handle hex-suffix-style names).
    for pat in _OUT_OF_SCOPE_PATTERNS:
        if pat.search(name):
            return False
    return True


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main():
    print("[1/4] Loading sources ...")
    all_records: list[Incident] = []
    for loader in (load_defillama, load_rekt, load_kismp,
                   load_defihacklabs, load_blocksec,
                   load_defi_rekt, load_slowmist):
        all_records.extend(loader())

    print("[2/4] Pre-merge date corrections + dedup ...")
    apply_date_corrections(all_records)
    merged = dedup_merge(all_records, date_tol_days=21)
    print(f"  pre-dedup raw count   : {len(all_records)}")
    print(f"  post-dedup unique     : {len(merged)}")

    print("[3/4] Filtering + tagging sector + SOA category ...")
    out_rows = []
    src_dist = defaultdict(int)
    other_dropped_count = 0
    other_dropped_usd = 0.0
    for r in merged:
        if r.loss_usd <= 0 or not looks_defi_protocol(r):
            continue
        sector = infer_sector(r.name, r.technique, r.description,
                              r.source_sector)
        basel  = infer_basel(r.name, r.technique, r.description,
                             r.classification)
        soa    = CHANG_FROM_BASEL[basel]
        # Round-3 expansion: only-if-already-Other description filter.
        # Events that infer_sector landed in Other AND whose
        # name+description matches one of the non-DeFi-protocol patterns
        # (NFT mint, pure memecoin, Ponzi, fundraising scam, KOL wallet
        # phishing, gambling, GameFi, SocialFi, CEX infrastructure) are
        # dropped. This filter is sector-gated so it cannot mis-drop a
        # legitimate Bridge / Lending / DEX / Yield / Stablecoin /
        # Derivatives event whose description happens to mention an
        # NFT-adjacent keyword.
        if sector == "Other":
            haystack = f"{r.name} {r.description or ''}"
            if any(p.search(haystack) for p in _OUT_OF_SCOPE_DESC_PATTERNS):
                other_dropped_count += 1
                other_dropped_usd += r.loss_usd
                continue
        out_rows.append({
            "date":             r.date.isoformat(),
            "name":             r.name,
            "loss_usd":         r.loss_usd,
            "recovered_usd":    r.recovered_usd,
            "net_usd":          max(0.0, r.loss_usd - r.recovered_usd),
            "chain":            r.chain,
            "sector":           sector,
            "basel2_category":  basel,
            "soa_category":     soa,
            "classification":   r.classification,
            "technique":        r.technique,
            "description":      r.description,
            "sources":          ",".join(sorted(r.sources)),
            "source_urls":      " | ".join(sorted(r.source_urls)),
            "n_sources":        len(r.sources),
        })
        src_dist[",".join(sorted(r.sources))] += 1
    print(f"  Other-sector description filter dropped {other_dropped_count} "
          f"events (USD ${other_dropped_usd/1e6:.1f}m)")

    df = (pd.DataFrame(out_rows)
            .sort_values("date")
            .reset_index(drop=True))

    # Freeze the study window: drop events after the cutoff so re-running the
    # loaders is reproducible (newer scrapes would otherwise extend the window).
    df = df[pd.to_datetime(df["date"]).dt.date <= ANALYSIS_WINDOW_END].copy()
    n_windowed = len(df)

    # Curated sector re-audit: correct mis-tagged DeFi sectors and drop non-DeFi
    # events (NOT_DEFI) so they never enter the consolidated dataset.
    df = apply_sector_reassignment(df)
    non_defi = df["sector"] == "NOT_DEFI"
    n_non_defi = int(non_defi.sum())
    df = (df[~non_defi]
            .sort_values("date")
            .reset_index(drop=True))

    fp = DATA / "events_consolidated.csv"
    df.to_csv(fp, index=False)

    print(f"  window-capped consolidated : {n_windowed}")
    print(f"  non-DeFi events dropped     : {n_non_defi}")
    print(f"\n[4/4] Wrote {len(df)} DeFi records to {fp}")
    print(f"  date range  : {df['date'].min()} .. {df['date'].max()}")
    print(f"  total gross : USD {df['loss_usd'].sum()/1e9:.2f} B")
    print(f"\n  source overlap:")
    for k, v in sorted(src_dist.items(), key=lambda x: -x[1]):
        print(f"    {k:<35s} : {v}")
    print(f"\n  Basel II category × n / sum:")
    by_basel = df.groupby("basel2_category").agg(
        n=("name", "count"), sum_usd=("loss_usd", "sum"),
        median_usd=("loss_usd", "median"))
    print(by_basel.sort_values("n", ascending=False).to_string())
    print(f"\n  Chang (SOA) category × n / sum:")
    by_cat = df.groupby("soa_category").agg(
        n=("name", "count"), sum_usd=("loss_usd", "sum"),
        median_usd=("loss_usd", "median"))
    print(by_cat.sort_values("n", ascending=False).to_string())
    print(f"\n  Sector × n / sum:")
    by_sec = df.groupby("sector").agg(
        n=("name", "count"), sum_usd=("loss_usd", "sum"),
        median_usd=("loss_usd", "median"))
    print(by_sec.sort_values("n", ascending=False).to_string())


if __name__ == "__main__":
    main()
