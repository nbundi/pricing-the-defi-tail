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
                r"project\s*owner\s*internal)\b", re.I),
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
                r"first[ -]?depositor|lack\s*of\s*liquidity)\b", re.I),
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


def infer_sector(name: str, technique: str, description: str,
                 default_sector: str) -> str:
    """Return a sector for the record. If default_sector is set
    (DefiLlama-tagged), it is honored after the standard folds:
    DefiLlama's ``Algo-Stables'' becomes our unified ``Stablecoin'',
    and RWA folds into ``Other'' (the RWA bucket is sparsely populated
    and conceptually overlaps with Stablecoin/Yield once tokenised-
    treasury issuers are split out)."""
    haystack = f"{name} {technique or ''} {description or ''}"
    if default_sector and default_sector not in ("Other", ""):
        return _SECTOR_FOLD.get(default_sector, default_sector)
    for pat, sec in SECTOR_RULES:
        if pat.search(haystack):
            return _SECTOR_FOLD.get(sec, sec)
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


def normalize_name(name: str) -> str:
    if not name:
        return ""
    s = _SUFFIX_STRIP.sub("", name.strip())
    s = re.sub(r"\([^)]*\)", "", s)         # drop parenthetical descriptions
    s = re.sub(r"\s+(v?\d+)$", "", s, flags=re.I)
    s = re.sub(r"[^a-zA-Z0-9]", "", s).lower()
    return s


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

def dedup_merge(records: list[Incident],
                date_tol_days: int = 14) -> list[Incident]:
    """Two-pass dedup: name-cluster + (date,amount)-cluster."""
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
                pass1.append(_merge_cluster(cluster)); cluster = [r]
        if cluster:
            pass1.append(_merge_cluster(cluster))

    # Pass 2: same-date(±7) AND amount within 10%
    pass1.sort(key=lambda r: (r.date, -r.loss_usd))
    merged: list[Incident] = []
    skip: set[int] = set()
    for i, r in enumerate(pass1):
        if i in skip:
            continue
        cluster = [r]
        for j in range(i + 1, len(pass1)):
            if j in skip:
                continue
            s = pass1[j]
            if (s.date - r.date).days > 7:
                break
            if abs(s.loss_usd - r.loss_usd) <= 0.10 * max(r.loss_usd,
                                                          s.loss_usd):
                cluster.append(s); skip.add(j)
        merged.append(_merge_cluster(cluster) if len(cluster) > 1 else r)
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


def looks_defi_protocol(r: Incident) -> bool:
    if r.defillama_known:
        return r.is_defi_protocol
    if NON_DEFI_PROTOCOL.search(r.name):
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
    merged = dedup_merge(all_records, date_tol_days=14)
    print(f"  pre-dedup raw count   : {len(all_records)}")
    print(f"  post-dedup unique     : {len(merged)}")

    print("[3/4] Filtering + tagging sector + SOA category ...")
    out_rows = []
    src_dist = defaultdict(int)
    for r in merged:
        if r.loss_usd <= 0 or not looks_defi_protocol(r):
            continue
        sector = infer_sector(r.name, r.technique, r.description,
                              r.source_sector)
        basel  = infer_basel(r.name, r.technique, r.description,
                             r.classification)
        soa    = CHANG_FROM_BASEL[basel]
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

    df = (pd.DataFrame(out_rows)
            .sort_values("date")
            .reset_index(drop=True))
    fp = DATA / "events_consolidated.csv"
    df.to_csv(fp, index=False)

    print(f"\n[4/4] Wrote {len(df)} records to {fp}")
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
