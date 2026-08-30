"""Quantitative risk analysis of DeFi exploit losses.

Inputs:  data/events_consolidated.csv  (produced by events_consolidation.py)
         data/sector_tvl_panel.csv

Outputs: figures/r*_*.png              (EVT / LDA figures used in the paper)
         output/risk_summary.json      (machine-readable headline numbers)

Methods applied:
    1. Severity tail analysis: Hill, mean-excess, POT-GPD MLE, and a
       Vuong non-nested test of the GPD against a lognormal alternative.
    2. Frequency model: monthly Poisson vs negative-binomial, dispersion test.
    3. Loss-distribution approach (LDA): compound Poisson-GPD Monte Carlo
       of annual aggregate losses per sector; VaR/ES at 95/99/99.5/99.9%.
    4. Protocol-level capital adequacy: top-10 Lending venues vs their
       on-chain safety reserves; aggregate gap headline number.
    5. Yield-side compensation: per-protocol stablecoin supply APY vs the
       risk-free rate and the per-sector LDA pure premium.
    6. Comparing the two responses: cross-venue risk-spread dispersion and
       a one-sided Mann-Whitney test of whether buffered venues pay a lower
       depositor yield spread than unbuffered ones (do the two substitute?).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats, optimize, special

# Embed real fonts in vector output; matplotlib's default Type 3 fonts
# are rejected by most publisher pipelines, Springer's included.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def _usd(x: float) -> str:
    """Format USD amount avoiding the `$` character (matplotlib math mode)."""
    if abs(x) >= 1e9:
        return f"USD {x/1e9:,.1f}B"
    if abs(x) >= 1e6:
        return f"USD {x/1e6:,.1f}m"
    if abs(x) >= 1e3:
        return f"USD {x/1e3:,.1f}k"
    return f"USD {x:,.0f}"

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
OUT = ROOT / "output"
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

RNG = np.random.default_rng(20260518)

# ---------------------------------------------------------------------------
# 0. Load data
# ---------------------------------------------------------------------------

# Analysis window. Raw data go back to 2011, but the analysis is
# restricted to 2020-01-01 onward, in line with Aramonte et al. (2021)'s
# dating of the DeFi-as-a-non-trivial-financial-substrate era to 2020-Q3
# (Compound governance liquidity mining and the AMM TVL expansion that
# followed). Every analysis, plot, and model fit in the paper uses this
# window.
ANALYSIS_WINDOW_START = pd.Timestamp("2020-01-01")
# End of the frozen study window. events_consolidated.csv is capped at this
# date upstream (see events_consolidation.ANALYSIS_WINDOW_END); the loaders
# enforce it again so a stale CSV can't silently extend the window.
ANALYSIS_WINDOW_END = pd.Timestamp("2026-05-29")


def load_hacks(full_window: bool = False) -> pd.DataFrame:
    """Load the canonical multi-source incident master CSV produced by
    events_consolidation.py. By default filters to ANALYSIS_WINDOW_START
    (2020-01-01) onward, which is the window used for every analysis,
    plot, and model fit in the paper. `full_window=True` is a debug
    option that retains the sparse pre-2020 records (8 events) — not
    used by the production pipeline."""
    fp = DATA / "events_consolidated.csv"
    if not fp.exists():
        raise FileNotFoundError(
            f"Canonical event dataset not found at {fp}. "
            f"Run `python events_consolidation.py` first.")
    h = pd.read_csv(fp, parse_dates=["date"])
    h = h.rename(columns={"loss_usd": "gross"})
    if "recovered_usd" in h.columns:
        h = h.rename(columns={"recovered_usd": "recovered",
                              "net_usd": "net"})
    else:
        h["recovered"] = 0.0
        h["net"] = h["gross"]
    mask = h["gross"] > 0
    if not full_window:
        mask &= (h["date"] >= ANALYSIS_WINDOW_START) & (h["date"] <= ANALYSIS_WINDOW_END)
    h = h[mask].copy()
    h["bridge_hack"] = (h["sector"] == "Bridge")
    h = h.sort_values("date").reset_index(drop=True)
    return h


def load_tvl_panel() -> pd.DataFrame:
    p = pd.read_csv(DATA / "sector_tvl_panel.csv", parse_dates=["date"])
    p = p.set_index("date")
    return p


# ---------------------------------------------------------------------------
# 1. Tail estimators
# ---------------------------------------------------------------------------

def hill_estimator(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hill (1975) tail-index estimator as a function of order statistics.

    For order statistics X_(1) >= X_(2) >= ... >= X_(n), the Hill estimator
    based on the top k+1 order statistics is

        xi_hat_H(k) = (1/k) * sum_{i=1..k} ln(X_(i)) - ln(X_(k+1))

    Returns (k_array, xi_hat_array) with k = 5 .. n//2.
    """
    xs = np.sort(x)[::-1]
    n = len(xs)
    lnxs = np.log(xs)
    ks = np.arange(5, n // 2 + 1)
    xis = np.array([lnxs[:k].mean() - lnxs[k] for k in ks])
    return ks, xis


def mean_excess(x: np.ndarray, n_thresh: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Mean excess function e(u) = E[X-u | X>u]. Linear in u for GPD."""
    xs = np.sort(x)
    n = len(xs)
    # threshold grid: quantile-spaced 50%..98%
    qs = np.linspace(0.05, 0.98, n_thresh)
    us = np.quantile(xs, qs)
    es = np.array([(xs[xs > u] - u).mean() if (xs > u).sum() >= 5 else np.nan
                   for u in us])
    return us, es


def gpd_mle(excesses: np.ndarray) -> tuple[float, float, float]:
    """MLE for GPD parameters (xi, beta). Returns (xi, beta, loglik)."""
    e = excesses[excesses > 0]
    if len(e) < 10:
        return np.nan, np.nan, np.nan

    def neg_ll(params):
        xi, beta = params
        if beta <= 0:
            return 1e10
        if abs(xi) < 1e-8:
            # exponential limit
            return len(e) * math.log(beta) + e.sum() / beta
        y = 1 + xi * e / beta
        if (y <= 0).any():
            return 1e10
        return len(e) * math.log(beta) + (1 + 1/xi) * np.log(y).sum()

    # multi-start to avoid local minima
    best = None
    for xi0 in (0.1, 0.5, 1.0):
        try:
            res = optimize.minimize(neg_ll, [xi0, e.mean()], method="Nelder-Mead",
                                    options={"xatol": 1e-7, "fatol": 1e-7,
                                             "maxiter": 5000})
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            pass
    if best is None:
        return np.nan, np.nan, np.nan
    xi, beta = best.x
    return float(xi), float(beta), float(-best.fun)


def fit_pot_gpd(x: np.ndarray, threshold_q: float = 0.90):
    """Peaks-over-threshold GPD fit. Returns dict with parameters."""
    u = float(np.quantile(x, threshold_q))
    exceed = x[x > u] - u
    xi, beta, ll = gpd_mle(exceed)
    return {"threshold_q": threshold_q, "threshold_usd": u,
            "n_exceedances": int(len(exceed)),
            "xi": xi, "beta": beta, "loglik": ll}


# ---------------------------------------------------------------------------
# Plateau-stability threshold rule (top-down "upper stable quantile")
#
# Replaces the legacy step-function rule with a per-sector heuristic that
# (i) targets the upper end of the contiguous stable plateau in xi-hat(q)
# and (ii) enforces both an absolute and a relative floor on the
# GPD-calibration exceedance count n_u:
#
#   Q = { q in {0.50, 0.55, ..., 0.90} :
#           n_u(q) >= max(n_u_abs, ceil(n_u_frac * n_sector)) }
#   q* = max { q in Q : | xi_hat(q) - xi_hat(q - 0.05) | <= tau }
#
# Defaults:
#   tau = 0.30  (stability tolerance on consecutive-step xi-hat change)
#   n_u_abs = 20  (EKM 1997 Sec. 6.4 practitioner floor)
#   n_u_frac = 0.10  (DuMouchel 1983 "top 10%" rule)
#
# If no stable q is found in Q, fall back to the lowest q in Q.
# ---------------------------------------------------------------------------
def select_threshold_plateau(x: np.ndarray, qs=None,
                             tau: float = 0.30,
                             n_u_abs: int = 20, n_u_frac: float = 0.10) -> dict:
    """Top-down stable-plateau threshold selector. Returns the chosen q*
    plus per-q diagnostics that the orchestrator and the appendix stability
    figure consume."""
    if qs is None:
        qs = np.round(np.arange(0.50, 0.91, 0.05), 2)
    n_sec = int(len(x))
    floor_rel = max(int(n_u_abs), int(np.ceil(n_u_frac * n_sec)))
    diag = []
    for q in qs:
        u = float(np.quantile(x, float(q)))
        excess = x[x > u] - u
        n_u = int(len(excess))
        if n_u >= 8:
            xi, beta, _ = gpd_mle(excess)
        else:
            xi, beta = float("nan"), float("nan")
        diag.append({"q": float(q), "u": u, "n_u": n_u,
                     "xi": xi, "beta": beta,
                     "eligible": (n_u >= floor_rel) and np.isfinite(xi)})
    eligible_idx = [i for i, d in enumerate(diag) if d["eligible"]]
    if not eligible_idx:
        # No q meets the floor — fall back to highest q with a finite xi.
        finite_idx = [i for i, d in enumerate(diag) if np.isfinite(d["xi"])]
        if not finite_idx:
            return {"q_star": float(qs[-1]), "fallback": "no_finite_fit",
                    "diag": diag, "floor_used": floor_rel,
                    "tau": tau, "n_sector": n_sec}
        i = finite_idx[-1]
        return {"q_star": float(qs[i]), "fallback": "no_q_meets_floor",
                "diag": diag, "floor_used": floor_rel,
                "tau": tau, "n_sector": n_sec}
    # Top-down: highest eligible q whose consecutive backward step is within tau.
    for i in reversed(eligible_idx):
        if i == 0 or not np.isfinite(diag[i - 1]["xi"]):
            continue
        if abs(diag[i]["xi"] - diag[i - 1]["xi"]) <= tau:
            return {"q_star": float(qs[i]),
                    "xi_at_q_star": float(diag[i]["xi"]),
                    "n_u_at_q_star": int(diag[i]["n_u"]),
                    "diag": diag, "floor_used": floor_rel,
                    "tau": tau, "n_sector": n_sec}
    # Fallback: lowest eligible q.
    i = eligible_idx[0]
    return {"q_star": float(qs[i]), "fallback": "no_stable_step",
            "xi_at_q_star": float(diag[i]["xi"]),
            "n_u_at_q_star": int(diag[i]["n_u"]),
            "diag": diag, "floor_used": floor_rel,
            "tau": tau, "n_sector": n_sec}


def bootstrap_pot_gpd(x: np.ndarray, threshold_q: float,
                      n_boot: int = 1000, rng=None) -> dict:
    """Parametric bootstrap of POT-GPD fit. Returns 95% CIs and IQR on (xi,
    beta), plus the bootstrap standard errors.

    Heavy-tail tail-index inference is one of the canonical settings in which
    the non-parametric bootstrap is inconsistent: when the tail signal is
    carried by O(1) extreme observations, ~37% of any resample omits any
    given event, and the resampled $\\hat\\xi$ distribution is systematically
    left-shifted relative to the original-sample MLE (Resnick 2007 §4.7;
    Politis-Romano-Wolf 1999 §11). The parametric variant simulates from
    the fitted GPD$(\\hat\\xi, \\hat\\beta)$ at the original threshold,
    preserving the tail behaviour at each replicate.
    """
    from scipy.stats import genpareto
    if rng is None:
        rng = np.random.default_rng(42)
    # Fit at threshold to get the parametric centre.
    u_orig = float(np.quantile(x, threshold_q))
    excess_orig = x[x > u_orig] - u_orig
    n_u = int(len(excess_orig))
    if n_u < 10:
        return {"method": "parametric",
                "n_bootstrap": 0, "n_u": n_u,
                "xi_lo": float("nan"), "xi_hi": float("nan"),
                "xi_se": float("nan"),
                "xi_iqr_lo": float("nan"), "xi_iqr_hi": float("nan"),
                "beta_lo": float("nan"), "beta_hi": float("nan"),
                "beta_se": float("nan")}
    xi_hat, beta_hat, _ = gpd_mle(excess_orig)
    if not (np.isfinite(xi_hat) and np.isfinite(beta_hat) and beta_hat > 0):
        return {"method": "parametric",
                "n_bootstrap": 0, "n_u": n_u,
                "xi_lo": float("nan"), "xi_hi": float("nan"),
                "xi_se": float("nan"),
                "xi_iqr_lo": float("nan"), "xi_iqr_hi": float("nan"),
                "beta_lo": float("nan"), "beta_hi": float("nan"),
                "beta_se": float("nan")}
    xis, betas = [], []
    for _ in range(n_boot):
        y_sim = genpareto.rvs(c=xi_hat, loc=0, scale=beta_hat,
                              size=n_u, random_state=rng)
        xi_b, beta_b, _ = gpd_mle(y_sim)
        if not (np.isfinite(xi_b) and np.isfinite(beta_b) and beta_b > 0):
            continue
        if -5 < xi_b < 10:
            xis.append(xi_b); betas.append(beta_b)
    xis = np.array(xis); betas = np.array(betas)
    if len(xis) < 50:
        return {"method": "parametric",
                "n_bootstrap": int(len(xis)), "n_u": n_u,
                "xi_lo": float("nan"), "xi_hi": float("nan"),
                "xi_se": float("nan"),
                "xi_iqr_lo": float("nan"), "xi_iqr_hi": float("nan"),
                "beta_lo": float("nan"), "beta_hi": float("nan"),
                "beta_se": float("nan")}
    return {
        "method": "parametric",
        "n_bootstrap": int(len(xis)),
        "n_u": n_u,
        "xi_lo": float(np.quantile(xis, 0.025)),
        "xi_hi": float(np.quantile(xis, 0.975)),
        "xi_iqr_lo": float(np.quantile(xis, 0.25)),
        "xi_iqr_hi": float(np.quantile(xis, 0.75)),
        "xi_se": float(xis.std(ddof=1)),
        "beta_lo": float(np.quantile(betas, 0.025)),
        "beta_hi": float(np.quantile(betas, 0.975)),
        "beta_se": float(betas.std(ddof=1)),
    }


def bootstrap_lda_bps(x: np.ndarray, threshold_q: float,
                       annual_lambda: float, tvl: float,
                       n_boot: int = 200, n_years: int = 50_000,
                       frequency: str = "poisson", nb_alpha: float | None = None,
                       cap: float | None = None,
                       rng=None) -> dict:
    """Non-parametric bootstrap of the LDA capital ratios. For each replicate:
    resample x, refit POT-GPD, build the (capped) severity sampler,
    Monte-Carlo the compound (Poisson or NB) annual aggregate, and record
    the mean and the VaR99.9 in bps. Returns 95% CIs and the IQR."""
    if rng is None:
        rng = np.random.default_rng(123)
    n = len(x)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb = x[idx]
        u = float(np.quantile(xb, threshold_q))
        excess = xb[xb > u] - u
        if len(excess) < 10:
            continue
        xi, beta, _ = gpd_mle(excess)
        if not (np.isfinite(xi) and np.isfinite(beta) and beta > 0):
            continue
        samp = severity_sampler_pot(xb, u, xi, beta, cap=cap)
        tot = lda_simulate(annual_lambda, samp, n_years=n_years, rng=rng,
                            frequency=frequency, nb_alpha=nb_alpha)
        var999 = float(np.quantile(tot, 0.999))
        rows.append((tot.mean() / tvl * 1e4, var999 / tvl * 1e4))
    if len(rows) < 30:
        return {"n_bootstrap": len(rows),
                "mean_bps_lo": float("nan"), "mean_bps_hi": float("nan"),
                "var999_bps_lo": float("nan"), "var999_bps_hi": float("nan")}
    arr = np.array(rows)
    v = arr[:, 1]
    return {
        "n_bootstrap": len(rows),
        "mean_bps_lo": float(np.quantile(arr[:, 0], 0.025)),
        "mean_bps_hi": float(np.quantile(arr[:, 0], 0.975)),
        "var999_bps_lo": float(np.quantile(v, 0.025)),
        "var999_bps_hi": float(np.quantile(v, 0.975)),
        "var999_bps_median": float(np.median(v)),
        "var999_bps_iqr_lo": float(np.quantile(v, 0.25)),
        "var999_bps_iqr_hi": float(np.quantile(v, 0.75)),
        "var999_bps_p16": float(np.quantile(v, 0.16)),
        "var999_bps_p84": float(np.quantile(v, 0.84)),
    }


# ---------------------------------------------------------------------------
# 2. Distribution fits (GPD vs lognormal, for the Vuong comparison)
# ---------------------------------------------------------------------------

def gpd_logpdf_x(x, u, xi, beta):
    """Pointwise log density IN x of the POT-GPD model, valid on x > u.

    The GPD is fitted to the excesses Y = X - u, so the conditional density
    of X given X > u is f_Y(x - u). Returning it in x (not in the excess)
    is what makes it directly comparable, point for point, with a lognormal
    truncated to the same half-line.
    """
    z = (np.asarray(x, dtype=float) - u) / beta
    if abs(xi) < 1e-10:
        return -math.log(beta) - z
    arg = 1.0 + xi * z
    if np.any(arg <= 0):
        return np.full_like(z, -np.inf)
    return -math.log(beta) - (1.0 / xi + 1.0) * np.log(arg)


def lognormal_trunc_logpdf_x(x, u, mu, sigma):
    """Pointwise log density in x of a lognormal TRUNCATED to (u, inf).

    Uses the survival function directly rather than 1 - cdf: at the high
    thresholds used here the complement loses most of its significant
    digits, and the renormalising constant is exactly where that matters.
    """
    sf = stats.lognorm.sf(u, s=sigma, scale=math.exp(mu))
    if sf <= 0:
        return np.full(np.shape(x), -np.inf)
    return stats.lognorm.logpdf(x, s=sigma, scale=math.exp(mu)) - math.log(sf)


def fit_lognormal_truncated(tail: np.ndarray, u: float):
    """MLE of (mu, sigma) for a lognormal truncated to (u, inf), fitted on
    the exceedances themselves.

    Fitting on the exceedances -- rather than fitting the full sector sample
    and renormalising afterwards -- is what puts the lognormal on the same
    estimation basis as the GPD, which only ever sees the exceedances. The
    two bases give opposite signs for the comparison (see the module note in
    vuong_gpd_lognormal), so this is not a cosmetic choice.
    """
    lt = np.log(tail)
    mu0, sd0 = float(lt.mean()), float(max(lt.std(ddof=0), 1e-3))

    def neg(p):
        mu, sigma = p[0], math.exp(p[1])
        if not np.isfinite(sigma) or sigma <= 0:
            return 1e12
        s = float(lognormal_trunc_logpdf_x(tail, u, mu, sigma).sum())
        return 1e12 if not np.isfinite(s) else -s

    best, best_val = None, np.inf
    # Multi-start: the truncated likelihood is flat along a mu/sigma ridge
    # when u sits far into the upper tail, so a single start can stop early.
    for m0 in (mu0, mu0 - 1.0, mu0 + 1.0, math.log(u)):
        for s0 in (sd0, sd0 * 0.5, sd0 * 2.0, 1.0):
            r = optimize.minimize(neg, np.array([m0, math.log(s0)]),
                                  method="Nelder-Mead",
                                  options={"maxiter": 5000, "xatol": 1e-9,
                                           "fatol": 1e-9})
            if r.fun < best_val:
                best_val, best = float(r.fun), r.x
    return {"mu": float(best[0]), "sigma": float(math.exp(best[1])),
            "loglik": float(-best_val)}


def vuong_gpd_lognormal(x: np.ndarray, u: float) -> dict:
    """Vuong (1989) non-nested LR test of POT-GPD against lognormal.

    Both models are conditioned on X > u, fitted by MLE to the same strict
    exceedance set, and evaluated as densities in x on (u, inf). Both carry
    two free parameters, so the Vuong dimension correction vanishes and the
    statistic is the plain normalised mean log-likelihood difference:

        d_i = log f_GPD(x_i) - log f_LN(x_i)
        V   = sqrt(n_u) * mean(d) / sd(d)   ->  N(0, 1) under H0

    V > 0 favours the GPD, V < 0 the lognormal; |V| > 1.96 separates them at
    5%. Note the sign is NOT robust to how the lognormal is estimated -- a
    lognormal fitted to the full sample and renormalised over (u, inf) flips
    it in six of seven sectors -- which is itself the finding: at these n_u
    the two models are not distinguishable, and any directional reading of
    the statistic is noise.
    """
    tail = np.sort(np.asarray(x, dtype=float)[np.asarray(x, dtype=float) > u])
    n_u = len(tail)
    if n_u < 10:
        return {"n_u": n_u, "V": float("nan"), "p": float("nan"),
                "ll_gpd": float("nan"), "ll_ln": float("nan")}
    xi, beta, _ = gpd_mle(tail - u)
    if not (np.isfinite(xi) and np.isfinite(beta) and beta > 0):
        return {"n_u": n_u, "V": float("nan"), "p": float("nan"),
                "ll_gpd": float("nan"), "ll_ln": float("nan")}
    lg = gpd_logpdf_x(tail, u, xi, beta)
    lnf = fit_lognormal_truncated(tail, u)
    ll = lognormal_trunc_logpdf_x(tail, u, lnf["mu"], lnf["sigma"])
    d = np.asarray(lg) - np.asarray(ll)
    sd = float(d.std(ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        return {"n_u": n_u, "V": float("nan"), "p": float("nan"),
                "ll_gpd": float(lg.sum()), "ll_ln": float(ll.sum())}
    V = math.sqrt(n_u) * float(d.mean()) / sd
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(V))))
    return {"n_u": n_u, "V": float(V), "p": p,
            "ll_gpd": float(lg.sum()), "ll_ln": float(ll.sum()),
            "xi": float(xi), "beta": float(beta),
            "mu": lnf["mu"], "sigma": lnf["sigma"]}


# ---------------------------------------------------------------------------
# 3. Frequency model
# ---------------------------------------------------------------------------

def monthly_counts(h: pd.DataFrame, start: str | None = None) -> pd.Series:
    """Monthly hack counts over [start, last]. If start is None, uses the
    first-to-last span of the passed DataFrame (the production pipeline
    passes the post-2020-filtered `h`, so the effective window is
    2020-01-01..last)."""
    if start is None:
        start = h["date"].min().strftime("%Y-%m-01")
    h2 = h[h["date"] >= start]
    counts = h2.set_index("date").resample("MS").size()
    full = pd.date_range(start, h["date"].max(), freq="MS")
    return counts.reindex(full, fill_value=0)


def dispersion_test(counts: np.ndarray) -> dict:
    mu = counts.mean()
    var = counts.var(ddof=1)
    # Cameron-Trivedi regression-based test simplified:
    # under H0: var = mu (Poisson); compute index of dispersion
    n = len(counts)
    D = var / mu
    # asymptotic test stat for overdispersion
    z = (var - mu) / math.sqrt(2 * mu ** 2 / (n - 1))
    p = 1 - stats.norm.cdf(z)
    return {"mean": float(mu), "var": float(var), "D": float(D),
            "z": float(z), "p_overdisp": float(p)}


def fit_negbin(counts: np.ndarray) -> dict:
    """MLE for NB(r, p) parameterization via mean mu and dispersion alpha=1/r.

    Variance = mu + alpha * mu^2.
    """
    mu0 = counts.mean()
    var0 = counts.var(ddof=1)
    alpha0 = max((var0 - mu0) / mu0 ** 2, 1e-3) if mu0 > 0 else 0.5

    def neg_ll(params):
        log_mu, log_alpha = params
        mu = math.exp(log_mu); alpha = math.exp(log_alpha)
        r = 1 / alpha
        p = r / (r + mu)
        return -stats.nbinom.logpmf(counts, r, p).sum()

    res = optimize.minimize(neg_ll, [math.log(mu0), math.log(alpha0)],
                            method="Nelder-Mead",
                            options={"xatol": 1e-7, "fatol": 1e-7,
                                     "maxiter": 5000})
    mu = math.exp(res.x[0]); alpha = math.exp(res.x[1])
    return {"mu": float(mu), "alpha": float(alpha), "loglik": float(-res.fun)}


# ---------------------------------------------------------------------------
# 4. Loss-distribution approach (LDA): compound Poisson-GPD
# ---------------------------------------------------------------------------

def lda_simulate(annual_lambda: float, severity_sampler, n_years: int = 200_000,
                 rng=None, frequency: str = "poisson",
                 nb_alpha: float | None = None) -> np.ndarray:
    """Compound Monte Carlo of annual aggregate losses.

    `frequency` selects the annual-count distribution:
      - "poisson":  N ~ Poisson(annual_lambda)
      - "nb":       N ~ NegBinomial(mean=annual_lambda, dispersion=nb_alpha)
                    where Var(N) = annual_lambda * (1 + nb_alpha * annual_lambda)
                    in the standard NB(r=1/alpha, p=r/(r+mu)) parameterisation.
    """
    if rng is None:
        rng = np.random.default_rng()
    if frequency == "poisson":
        Ns = rng.poisson(annual_lambda, size=n_years)
    elif frequency == "nb":
        if nb_alpha is None or nb_alpha <= 0:
            raise ValueError("nb_alpha must be > 0 for NB frequency")
        r = 1.0 / nb_alpha
        p = r / (r + annual_lambda)
        Ns = rng.negative_binomial(r, p, size=n_years)
    else:
        raise ValueError(f"unknown frequency: {frequency}")
    totals = np.zeros(n_years)
    flat_N = int(Ns.sum())
    if flat_N == 0:
        return totals
    sev = severity_sampler(flat_N, rng)
    idx = 0
    for i, n in enumerate(Ns):
        if n == 0:
            continue
        totals[i] = sev[idx:idx + n].sum()
        idx += n
    return totals


def kupiec_pof_test(n_exceedances: int, n_observations: int,
                    confidence: float = 0.99) -> dict:
    """Kupiec proportion-of-failures (unconditional coverage) likelihood-
    ratio test. H0: realised exceedance rate equals the model's tail
    probability (1 - confidence).

    LR_uc = -2 * log(L0 / L1), chi-squared(1) under H0.
    """
    p = 1.0 - confidence
    x = n_exceedances
    n = n_observations
    if n == 0:
        return {"n_exc": 0, "n_obs": 0, "p_exc_realised": float("nan"),
                "p_exc_expected": p, "lr_uc": float("nan"), "p_value": float("nan")}
    pi_hat = x / n
    eps = 1e-12
    # log-likelihood under H0
    l0 = x * math.log(p + eps) + (n - x) * math.log(1 - p + eps)
    # log-likelihood under H1 (unrestricted)
    if 0 < pi_hat < 1:
        l1 = x * math.log(pi_hat) + (n - x) * math.log(1 - pi_hat)
    else:
        l1 = 0.0
    lr = -2 * (l0 - l1)
    pval = 1 - stats.chi2.cdf(lr, df=1)
    return {"n_exc": int(x), "n_obs": int(n),
            "p_exc_realised": float(pi_hat), "p_exc_expected": float(p),
            "lr_uc": float(lr), "p_value": float(pval)}


LST_LRT_PATTERN = (
    r"\b(?:Kelp|Lido|Rocket\s*Pool|EigenLayer|Renzo|Bedrock|uniBTC|"
    r"Ankr|StakeWise|Stader|sfrxETH|Frax\s*Ether|wstETH|rsETH|"
    r"ezETH|stETH|weETH|cbETH|ETHx)\b"
)


def sensitivity_fits(h: pd.DataFrame, fits: dict, rng=None) -> dict:
    """R6+R8 sensitivity re-fits requested in the pre-submission peer review.

    R6: Bridge tail with Kelp reclassified to LST/LRT; Derivatives split into
        Perps/Options vs LST/LRT.
    R8: Lending tail excluding events on small/mid-cap protocols (approximated
        as bottom 30% by gross-loss percentile, a proxy for protocol size at
        event time since per-protocol-TVL history is not available in the
        current panel).

    Returns a dict of sensitivity-fit summaries.
    """
    if rng is None:
        rng = np.random.default_rng(7)
    out = {}

    # --- R6: Derivatives split (Perps/Options vs LST/LRT) -----------------
    deriv = h.loc[h["sector"] == "Derivatives"].copy()
    is_lst = deriv["name"].str.contains(LST_LRT_PATTERN, case=False,
                                          regex=True, na=False)
    perps = deriv.loc[~is_lst, "gross"].values
    lst_in_deriv = deriv.loc[is_lst, "gross"].values
    if len(perps) >= 30:
        q0 = 0.70 if len(perps) < 80 else 0.80
        fit = fit_pot_gpd(perps, threshold_q=q0)
        boot = bootstrap_pot_gpd(perps, q0, n_boot=1000, rng=rng)
        out["derivatives_perps_only"] = {
            "n": int(len(perps)), "q": q0,
            "threshold_usd": fit["threshold_usd"],
            "xi": fit["xi"], "beta": fit["beta"],
            "xi_ci": [boot["xi_lo"], boot["xi_hi"]],
        }
    out["derivatives_lst_count"] = int(len(lst_in_deriv))
    out["derivatives_lst_sum_usd"] = float(lst_in_deriv.sum()) if len(lst_in_deriv) else 0.0

    # --- R8: Lending excluding bottom-30% by event-loss-percentile ---------
    # The domain reviewer requested excluding events on small/mid-cap
    # protocols at event time. Since per-protocol-TVL history is not in the
    # current panel, we approximate "small protocol" by the bottom 30% of
    # event losses, on the assumption that very small protocol-events
    # correlate with very small protocols. This is a coarse proxy.
    lending = h.loc[h["sector"] == "Lending", "gross"].values
    cutoff = float(np.quantile(lending, 0.30))
    lending_blue_chip = lending[lending > cutoff]
    if len(lending_blue_chip) >= 30:
        n_loc = len(lending_blue_chip)
        q0 = 0.80 if n_loc < 200 else 0.85
        fit = fit_pot_gpd(lending_blue_chip, threshold_q=q0)
        boot = bootstrap_pot_gpd(lending_blue_chip, q0, n_boot=1000, rng=rng)
        out["lending_ex_bottom30_loss_pctile"] = {
            "n": int(n_loc),
            "cutoff_usd": cutoff,
            "q": q0,
            "threshold_usd": fit["threshold_usd"],
            "xi": fit["xi"], "beta": fit["beta"],
            "xi_ci": [boot["xi_lo"], boot["xi_hi"]],
        }

    # Also: Lending excluding the two largest IF events (Vires + BXH) that
    # the domain reviewer specifically flagged as inflating the tail.
    lending_df = h.loc[h["sector"] == "Lending"].copy()
    flagged = lending_df["name"].str.contains(
        r"Vires|Boy\s*X\s*Highspeed|BXH",
        case=False, regex=True, na=False)
    lending_ex_flagged = lending_df.loc[~flagged, "gross"].values
    if len(lending_ex_flagged) >= 30:
        n_loc = len(lending_ex_flagged)
        q0 = 0.85
        fit = fit_pot_gpd(lending_ex_flagged, threshold_q=q0)
        boot = bootstrap_pot_gpd(lending_ex_flagged, q0, n_boot=1000, rng=rng)
        out["lending_ex_vires_bxh"] = {
            "n": int(n_loc), "q": q0,
            "threshold_usd": fit["threshold_usd"],
            "xi": fit["xi"], "beta": fit["beta"],
            "xi_ci": [boot["xi_lo"], boot["xi_hi"]],
        }

    return out


def out_of_sample_backtest(h: pd.DataFrame, sectors: list[str],
                            tvl_panel: pd.DataFrame,
                            fit_end: str = "2023-12-31",
                            test_end: str | None = None,
                            n_years_sim: int = 100_000,
                            nb_alpha: float | None = None,
                            rng=None) -> dict:
    """Refit per-sector POT-GPD + compound-Poisson LDA on `h` restricted to
    date <= fit_end, then compare the model's quantile predictions to the
    realised annual aggregate losses on (fit_end, test_end].

    For each sector with sufficient data on both windows, we report:
      - n_test_years: number of full-year buckets in the test window
      - es99_bps_in_sample: model-predicted ES99 (bps of fit-window TVL)
      - exc99 / exc999: realised annual exceedances at the 99% and 99.9%
                       VaR thresholds derived from the fit-window LDA
      - kupiec p-values at both confidence levels
    """
    if rng is None:
        rng = np.random.default_rng(2024)
    fit_dt = pd.Timestamp(fit_end)
    if test_end is None:
        test_dt = h["date"].max()
    else:
        test_dt = pd.Timestamp(test_end)
    h_fit = h[h["date"] <= fit_dt].copy()
    h_test = h[(h["date"] > fit_dt) & (h["date"] <= test_dt)].copy()
    years_fit = max((h_fit["date"].max() - h_fit["date"].min()).days / 365.25,
                    1.0) if len(h_fit) > 0 else 0.0
    out = {"fit_end": str(fit_dt.date()), "test_end": str(test_dt.date()),
            "years_fit": float(years_fit),
            "by_sector": {}}
    # Realised aggregate losses by sector and calendar year (test window):
    if len(h_test) > 0:
        h_test["year"] = h_test["date"].dt.year
        agg = (h_test.groupby(["sector", "year"])["gross"]
                       .sum().reset_index())
    else:
        agg = pd.DataFrame({"sector": [], "year": [], "gross": []})
    test_years = sorted(set(agg["year"].astype(int)))
    for sect in sectors:
        x_fit = h_fit.loc[h_fit["sector"] == sect, "gross"].values
        if len(x_fit) < 30:
            continue
        n_loc = len(x_fit)
        if n_loc >= 200:    q0 = 0.85
        elif n_loc >= 80:   q0 = 0.80
        elif n_loc >= 40:   q0 = 0.70
        else:               q0 = 0.60
        pot_s = fit_pot_gpd(x_fit, threshold_q=q0)
        if not np.isfinite(pot_s["xi"]):
            continue
        lam_s = len(x_fit) / years_fit
        try:
            tvl_s = float(tvl_panel[sect].iloc[-365:].mean())
        except KeyError:
            tvl_s = float("nan")
        samp = severity_sampler_pot(x_fit, pot_s["threshold_usd"],
                                     pot_s["xi"], pot_s["beta"])
        if nb_alpha is None:
            tot = lda_simulate(lam_s, samp, n_years=n_years_sim, rng=rng)
        else:
            tot = lda_simulate(lam_s, samp, n_years=n_years_sim, rng=rng,
                                frequency="nb", nb_alpha=nb_alpha)
        q99 = float(np.quantile(tot, 0.99))
        q999 = float(np.quantile(tot, 0.999))
        es99 = float(tot[tot >= q99].mean())
        # Realised annual aggregates in test window:
        sect_agg = agg.loc[agg["sector"] == sect].set_index("year")["gross"]
        realised = [float(sect_agg.get(y, 0.0)) for y in test_years]
        exc99 = sum(1 for r in realised if r > q99)
        exc999 = sum(1 for r in realised if r > q999)
        out["by_sector"][sect] = {
            "n_fit": int(n_loc),
            "lambda_yr_fit": float(lam_s),
            "xi_fit": float(pot_s["xi"]),
            "tvl_recent_usd": tvl_s,
            "var99_usd": q99,
            "var999_usd": q999,
            "es99_usd": es99,
            "es99_bps_fit_window_tvl": float(es99 / tvl_s * 1e4) if tvl_s else float("nan"),
            "realised_annual_losses_usd": realised,
            "test_years": test_years,
            "kupiec_99": kupiec_pof_test(exc99, len(test_years), 0.99),
            "kupiec_999": kupiec_pof_test(exc999, len(test_years), 0.999),
        }
    return out


def severity_sampler_pot(body_x: np.ndarray, threshold: float,
                         xi: float, beta: float, cap: float | None = None):
    """Mixture sampler: with prob p_below sample from empirical CDF below
    threshold (bootstrap), else sample from GPD(xi, beta) + threshold.
    If `cap` is set, every simulated loss is capped at `cap` (the largest
    single-protocol exposure in the sector): a single operational event
    cannot lose more than the assets at risk in one protocol."""
    below = body_x[body_x <= threshold]
    p_below = len(below) / len(body_x)

    def _sample(N, rng):
        out = np.empty(N)
        u = rng.random(N)
        is_tail = u >= p_below
        n_tail = is_tail.sum()
        if n_tail > 0:
            ut = rng.random(n_tail)
            # GPD inverse CDF: u -> threshold + beta/xi * ((1-u)^(-xi) - 1)
            if abs(xi) < 1e-8:
                tail = threshold + beta * (-np.log(1 - ut))
            else:
                tail = threshold + beta / xi * ((1 - ut) ** (-xi) - 1)
            out[is_tail] = tail
        n_body = N - n_tail
        if n_body > 0:
            out[~is_tail] = rng.choice(below, size=n_body, replace=True)
        if cap is not None:
            out = np.minimum(out, cap)
        return out
    return _sample


# ---------------------------------------------------------------------------
# 5. Protocol-level adequacy
# ---------------------------------------------------------------------------

def load_defillama_yields(fp: Path | None = None) -> pd.DataFrame:
    """Load the cached DefiLlama yields panel (stablecoin-supply + LP pools
    with TVL >= USD 1m). Each row is a pool with chain, project, symbol,
    tvlUsd, apyBase (fee revenue), apyReward (token incentives), and the
    total apy = base + reward."""
    if fp is None:
        fp = DATA / "raw" / "defillama" / "yields_pools.json"
    rows = json.loads(fp.read_text())
    df = pd.DataFrame(rows)
    return df


def per_protocol_supply_apy(yields_df: pd.DataFrame,
                              project_slugs: dict[str, str],
                              stablecoin_only: bool = True,
                              single_asset_only: bool = True) -> pd.DataFrame:
    """For each DefiLlama project slug, compute the TVL-weighted supply APY
    across the protocol's supply pools.

    The headline `apy_total_pct` is the trailing 30-day mean total APY
    (DefiLlama `apyMean30d`, base + reward), not a single-day spot rate:
    spot supply yields are volatile, and a depositor's realised
    compensation is closer to the trailing mean. Spot base/reward are
    retained for reference. `stablecoin_only=True` restricts to
    stablecoin-denominated pools; `single_asset_only=True` to single-asset
    pools (use False for DEX LP positions, which are multi-asset).
    `project_slugs` maps DefiLlama project slug -> display name.
    """
    df = yields_df.copy()
    if stablecoin_only:
        df = df[df["stablecoin"] == True]
    if single_asset_only:
        df = df[df["exposure"] == "single"]
    df = df[df["tvlUsd"] >= 1e6]
    df["project_lc"] = df["project"].fillna("").str.lower()
    # Trailing 30-day mean total APY per pool (base+reward), fall back to spot.
    df = df.copy()
    df["mean30"] = df["apyMean30d"].where(df["apyMean30d"].notna(),
                                          df["apyBase"].fillna(0) + df["apyReward"].fillna(0))
    # Exclude pools that pay no lending supply yield (mean30 <= 0.01%): these
    # are yield-bearing-collateral positions (sUSDe, syrupUSDC, sUSDS, USTB,
    # ...) supplied at ~0% lending APY, whose return accrues in the token, not
    # the market. Counting them as 0% understates the supply-yield premium.
    df = df[df["mean30"] > 0.01]
    out = []
    for slug, name in project_slugs.items():
        m = df[df["project_lc"] == slug]
        if len(m) == 0:
            out.append({"name": name, "slug": slug, "tvl_usd": 0.0,
                         "apy_base_pct": float("nan"),
                         "apy_reward_pct": float("nan"),
                         "apy_total_pct": float("nan"), "n_pools": 0})
            continue
        tvl = float(m["tvlUsd"].sum())
        base = float((m["apyBase"].fillna(0) * m["tvlUsd"]).sum() / tvl)
        rew  = float((m["apyReward"].fillna(0) * m["tvlUsd"]).sum() / tvl)
        total = float((m["mean30"] * m["tvlUsd"]).sum() / tvl)
        out.append({
            "name": name, "slug": slug, "tvl_usd": tvl,
            "apy_base_pct": base,
            "apy_reward_pct": rew,
            "apy_total_pct": total,
            "n_pools": int(len(m)),
        })
    return pd.DataFrame(out)


def yield_compensation_table(yields_df: pd.DataFrame,
                                sector_pure_premium_bps: dict[str, float],
                                tbill_pct: float = 3.70) -> dict:
    """Produce the cross-sector yield-vs-OpRisk-compensation table.

    For each sector with both an LDA pure premium and a defensible
    stablecoin-supply yield panel, compute:
      - Per-protocol TVL-weighted stablecoin base and total APY
      - Excess yield over T-bill (base-only and total)
      - Comparison to the per-sector pure premium (E[S]/TVL bps from LDA)

    `sector_pure_premium_bps` maps sector name -> pure premium in bps
    (e.g. {"Lending": 48.0, "DEX": 185.0, "Yield": 178.0}).
    """
    out = {"tbill_pct": float(tbill_pct), "by_sector": {}}

    # --- Lending: top-10 venues by current TVL (matches Table 12) ---------
    lending_slugs = {
        "aave-v3":           "Aave V3",
        "morpho-blue":       "Morpho Blue",
        "sparklend":         "SparkLend",
        "justlend-v1":       "JustLend V1",
        "kamino-lend":       "Kamino Lend",
        "compound-v3":       "Compound V3",
        "venus-core-pool":   "Venus Core Pool",
        "jupiter-lend":      "Jupiter Lend",
        "fluid-lending":     "Fluid Lending",
        "euler-v2":          "Euler V2",
    }
    lending = per_protocol_supply_apy(yields_df, lending_slugs,
                                        stablecoin_only=True)
    lending = lending.sort_values("tvl_usd", ascending=False).reset_index(drop=True)
    pp_lending = sector_pure_premium_bps.get("Lending", 0.0) / 100  # bps -> %
    out["by_sector"]["Lending"] = {
        "pure_premium_pct": float(pp_lending),
        "protocols": lending.to_dict(orient="records"),
        "tvl_weighted_base_pct": float(
            (lending["apy_base_pct"].fillna(0) * lending["tvl_usd"]).sum()
            / lending["tvl_usd"].sum()) if lending["tvl_usd"].sum() > 0 else float("nan"),
        "tvl_weighted_total_pct": float(
            (lending["apy_total_pct"].fillna(0) * lending["tvl_usd"]).sum()
            / lending["tvl_usd"].sum()) if lending["tvl_usd"].sum() > 0 else float("nan"),
    }

    # --- DEX: top stable-pool LP positions (Curve, Uniswap, Balancer) -----
    # DEX yield mechanics are different from lending (LP fees + IL + token
    # incentives), but the cleanest cross-section is stable-pool LP yields,
    # which carry minimal IL and are the closest analogue to "passive
    # deposit" in the DEX bucket.
    dex_slugs = {
        "curve-dex":       "Curve",
        "uniswap-v3":      "Uniswap V3",
        "uniswap-v4":      "Uniswap V4",
        "balancer-v2":     "Balancer V2",
        "balancer-v3":     "Balancer V3",
        "fluid-dex":       "Fluid DEX",
    }
    dex = per_protocol_supply_apy(yields_df, dex_slugs,
                                    stablecoin_only=True,
                                    single_asset_only=False)
    dex = dex.sort_values("tvl_usd", ascending=False).reset_index(drop=True)
    pp_dex = sector_pure_premium_bps.get("DEX", 0.0) / 100
    out["by_sector"]["DEX"] = {
        "pure_premium_pct": float(pp_dex),
        "protocols": dex.to_dict(orient="records"),
        "tvl_weighted_base_pct": float(
            (dex["apy_base_pct"].fillna(0) * dex["tvl_usd"]).sum()
            / dex["tvl_usd"].sum()) if dex["tvl_usd"].sum() > 0 else float("nan"),
        "tvl_weighted_total_pct": float(
            (dex["apy_total_pct"].fillna(0) * dex["tvl_usd"]).sum()
            / dex["tvl_usd"].sum()) if dex["tvl_usd"].sum() > 0 else float("nan"),
    }

    # --- Yield aggregators: stable vault APYs -----------------------------
    yield_slugs = {
        "yearn-finance":   "Yearn",
        "pendle":          "Pendle",
        "convex-finance":  "Convex",
        "spark":           "Spark sUSDS",
        "ethena":          "Ethena sUSDe",
        "morpho":          "Morpho Vaults",
        "harvest-finance": "Harvest",
    }
    yld = per_protocol_supply_apy(yields_df, yield_slugs,
                                    stablecoin_only=True,
                                    single_asset_only=False)
    yld = yld.sort_values("tvl_usd", ascending=False).reset_index(drop=True)
    pp_yield = sector_pure_premium_bps.get("Yield", 0.0) / 100
    out["by_sector"]["Yield"] = {
        "pure_premium_pct": float(pp_yield),
        "protocols": yld.to_dict(orient="records"),
        "tvl_weighted_base_pct": float(
            (yld["apy_base_pct"].fillna(0) * yld["tvl_usd"]).sum()
            / yld["tvl_usd"].sum()) if yld["tvl_usd"].sum() > 0 else float("nan"),
        "tvl_weighted_total_pct": float(
            (yld["apy_total_pct"].fillna(0) * yld["tvl_usd"]).sum()
            / yld["tvl_usd"].sum()) if yld["tvl_usd"].sum() > 0 else float("nan"),
    }

    return out


def top_n_lending_adequacy(
    n: int,
    gross_tvl_path: Path,
    sector_fit: dict,
    exclude: tuple = ("Maple", "Sky Lending", "USDD"),
    n_years: int = 1_000_000,
    rng=None,
) -> list[dict]:
    """Top-`n` Lending protocols by current GROSS supplied TVL
    (idle + borrowed; DefiLlama, 2026-06-30) with their disclosed
    operational-risk reserves and the modelled per-protocol capital
    targets (pure premium E[S] and regulatory capital VaR99.9).

    Allocation follows the sector thinning model but respects the
    non-linearity of the tail. Under the homogeneity assumption the
    sector event process thins to protocol p with rate
    lambda_p = (TVL_p / TVL_s) * lambda_s and the same per-event
    severity, capped at the protocol's OWN exposure (a single event
    cannot lose more than the assets held in one protocol). Both the pure
    premium E[S]_p and VaR99.9 are taken from that one simulated compound,
    so both respect that cap. Were severity uncapped the mean would thin
    exactly linearly and E[S]_p/TVL_p would equal the sector ratio for
    every venue; the cap breaks that invariance, and E[S]/TVL falls with
    venue size because a smaller venue truncates more of the severity
    distribution. The 99.9% VaR of a heavy-tailed
    compound is NOT linear in TVL: by the single-loss/subexponential
    result (Bocker & Kluppelberg 2005; Embrechts, Kluppelberg & Mikosch
    1997) VaR_alpha(S_p) ~ F^{-1}(1 - (1-alpha)/lambda_p), which for a
    regularly-varying tail scales as TVL_p^xi with xi < 1, so the
    VaR/TVL ratio rises as a venue shrinks. VaR99.9 is therefore
    simulated per protocol here rather than scaled by the sector ratio;
    the sector-ratio (linear) rule understates the smaller venues.

    `sector_fit` carries the once-fitted sector severity and frequency:
    keys `losses` (np.ndarray of sector event losses), `threshold_usd`,
    `xi`, `beta`, `lam_s` (annual sector rate), `alpha_annual` (NB
    dispersion, monthly/12), and `tvl_s` (sector TVL denominator).

    Basis: NET (idle) TVL is the funds actually at risk and the single
    consistent denominator across the paper, so thinning, the severity
    cap, E[S], and VaR all use each protocol's net TVL. Gross supplied
    TVL (net + active loans) is used ONLY to rank and select the top-10,
    so high-utilisation money markets (e.g. Euler) are not understated in
    the sample. Non-money-markets are excluded (reclassified `Other`):
    Maple (undercollateralised institutional credit), Sky Lending (an
    sDAI savings product, no borrow market), and USDD (a stablecoin).
    """
    if rng is None:
        rng = np.random.default_rng(20260701)
    recs = json.loads(Path(gross_tvl_path).read_text())
    recs = [r for r in recs
            if r.get("name") not in exclude and r.get("gross_usd")]
    recs.sort(key=lambda r: r["gross_usd"], reverse=True)
    top = recs[:n]

    # Protocol-side operational-risk reserves, verified against each
    # protocol's public safety-reserve dashboard or on-chain contract
    # (June 2026; source in each note). These are upper bounds where the
    # reserve is held in volatile or only-partly-slashable assets.
    BUFFER_USD: dict[str, tuple[float, str]] = {
        "Aave V3":         (0.363e9, "Safety Module + Umbrella, total staked "
                                    "(app.aave.com/safety-module, /staking)"),
        "Morpho Blue":     (0.00,   "no protocol-wide buffer; risk is "
                                    "per-market and borne by depositors"),
        "SparkLend":       (0.083e9, "Sky aggregate backstop capital "
                                    "(info.skyeco.com/capital-backstop)"),
        "JustLend V1":     (0.00,   "no public protocol-side reserve"),
        "Kamino Lend":     (0.00,   "no public protocol-wide buffer"),
        "Compound V3":     (0.043e9, "V3 reserves, Aera + in-Comet "
                                    "(compound.woof.software treasury; "
                                    "docs.compound.finance/liquidation)"),
        "Venus Core Pool": (0.0108e9, "Risk Fund, all assets in riskFund "
                                    "contract (BscScan 0xdF31...76E42)"),
        "Jupiter Lend":    (0.00,   "no public protocol-side reserve"),
        "Fluid Lending":   (0.00,   "no public protocol-side reserve"),
        "Euler V2":        (0.00,   "no public protocol-wide operational-risk "
                                    "reserve after the 2023 relaunch"),
        "HyperLend Pooled":(0.00,   "no public protocol-side reserve"),
        "Lista Lending":   (0.00,   "no public protocol-side reserve"),
    }

    losses_s   = np.asarray(sector_fit["losses"], dtype=float)
    thr_s      = float(sector_fit["threshold_usd"])
    xi_s       = float(sector_fit["xi"])
    beta_s     = float(sector_fit["beta"])
    lam_s      = float(sector_fit["lam_s"])
    alpha_ann  = float(sector_fit["alpha_annual"])
    tvl_s      = float(sector_fit["tvl_s"])

    out = []
    for r in top:
        name = r["name"]
        tvl_gross = float(r["gross_usd"])            # selection basis only
        tvl       = float(r["net_usd"])              # net = funds at risk (exposure)
        # Thin the sector process to protocol p on the net (at-risk) basis:
        # lower frequency, severity capped at p's own net exposure, VaR as a
        # fraction of net. tvl_s is the sector NET TVL (consistent basis).
        share  = tvl / tvl_s
        lam_p  = share * lam_s
        samp_p = severity_sampler_pot(losses_s, thr_s, xi_s, beta_s, cap=tvl)
        tot_p  = lda_simulate(lam_p, samp_p, n_years=n_years, rng=rng,
                              frequency="nb", nb_alpha=alpha_ann)
        var999 = float(np.quantile(tot_p, 0.999))
        var99  = float(np.quantile(tot_p, 0.99))
        var995 = float(np.quantile(tot_p, 0.995))
        # E[S] and VaR are both read off the SAME simulated compound, so both
        # carry the same per-protocol severity cap. Scaling the sector pure
        # premium linearly instead (E[S]_s/TVL_s * TVL_p) would be exact only
        # for UNCAPPED severity, and would report an identical bps figure for
        # every venue while the VaR column varies -- two different severity
        # assumptions in adjacent columns of the same table.
        meanS  = float(tot_p.mean())
        buffer, note = BUFFER_USD.get(name, (0.0, "no public protocol-side reserve"))
        out.append({
            "name": name,
            "tvl_usd": tvl,                # net (exposure)
            "gross_tvl_usd": tvl_gross,    # gross (selection only)
            "net_tvl_usd": tvl,
            "thinned_lambda_yr": lam_p,
            "var999_usd": var999,
            "var999_bps": var999 / tvl * 1e4,
            "var99_usd":  var99,
            "var995_usd": var995,
            "mean_usd": meanS,
            "buffer_usd": buffer,
            "buffer_note": note,
            "coverage":      (buffer / var999) if var999 > 0 else float("nan"),
            "coverage_99":   (buffer / var99)  if var99  > 0 else float("nan"),
            "coverage_995":  (buffer / var995) if var995 > 0 else float("nan"),
            "coverage_mean": (buffer / meanS)  if meanS  > 0 else float("nan"),
            "gap_usd": max(var999 - buffer, 0.0),
        })
    return out


def compare_responses(top10: list[dict],
                      lending_yield_protocols: list[dict],
                      tbill_pct: float = 3.70) -> dict:
    """Compare the two participant responses to DeFi operational risk
    across the top-10 Lending venues:

      Response 1 = the protocol capital buffer (coverage of the modelled
                   ES99 from `top_n_lending_adequacy`);
      Response 2 = the depositor yield spread (total stablecoin supply
                   APY over the risk-free rate, from
                   `yield_compensation_table`).

    Joins the two per-protocol tables by venue name and computes:
      - the cross-venue risk-spread dispersion (min/max/range/std, bps);
      - the mean risk spread for zero-buffer vs buffered venues, and the
        mean buffer coverage among buffered venues, with a one-sided
        Mann-Whitney test that buffered venues pay a lower spread.

    These are the statistics reported in the paper's "Comparing the two
    responses" section (Section~\\ref{sec:compare}).
    """
    apy_by_name = {p["name"]: p.get("apy_total_pct")
                   for p in lending_yield_protocols}
    rows = []
    for p in top10:
        apy = apy_by_name.get(p["name"])
        if apy is None or (isinstance(apy, float) and math.isnan(apy)):
            continue
        cov = p.get("coverage")
        cov = float(cov) if cov == cov else 0.0   # NaN coverage -> 0
        spread = float(apy) - tbill_pct           # risk spread over T-bill, %
        rows.append({
            "name":            p["name"],
            "tvl_usd":         float(p["tvl_usd"]),
            "buffer_usd":      float(p["buffer_usd"]),
            "coverage":        cov,
            "apy_total_pct":   float(apy),
            "risk_spread_pct": spread,
            "has_buffer":      bool(p["buffer_usd"] > 0),
        })

    if len(rows) < 3:
        return {"n": len(rows), "venues": rows}

    spreads = np.array([r["risk_spread_pct"] for r in rows])

    zero_buf = [r["risk_spread_pct"] for r in rows if not r["has_buffer"]]
    buffered = [r["risk_spread_pct"] for r in rows if r["has_buffer"]]
    buf_cov  = [r["coverage"]        for r in rows if r["has_buffer"]]

    # Two-group test: do buffered venues pay a lower risk premium than
    # unbuffered ones? A Mann-Whitney U (one-sided) is robust to the small
    # samples and to any single outlier, and tests the substitution
    # hypothesis directly. (A rank correlation over all venues is not
    # reported: the top-10 sample is too small to support it.)
    if buffered and zero_buf:
        mw_u, mw_p = stats.mannwhitneyu(buffered, zero_buf, alternative="less")
    else:
        mw_u, mw_p = float("nan"), float("nan")

    return {
        "n":                               len(rows),
        "tbill_pct":                       float(tbill_pct),
        "venues":                          rows,
        "risk_spread_min_bps":             float(spreads.min() * 100),
        "risk_spread_max_bps":             float(spreads.max() * 100),
        "risk_spread_range_bps":           float((spreads.max() - spreads.min()) * 100),
        "risk_spread_std_bps":             float(spreads.std(ddof=0) * 100),
        "n_zero_buffer":                   len(zero_buf),
        "n_buffered":                      len(buffered),
        "mean_spread_zero_buffer_bps":     float(np.mean(zero_buf) * 100) if zero_buf else float("nan"),
        "mean_spread_buffered_bps":        float(np.mean(buffered) * 100) if buffered else float("nan"),
        "median_spread_zero_buffer_bps":   float(np.median(zero_buf) * 100) if zero_buf else float("nan"),
        "median_spread_buffered_bps":      float(np.median(buffered) * 100) if buffered else float("nan"),
        "mannwhitney_u":                   float(mw_u),
        "mannwhitney_p_onesided":          float(mw_p),
        "mean_coverage_buffered":          float(np.mean(buf_cov)) if buf_cov else float("nan"),
    }


# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------

PALETTE = {
    # Sector
    "Lending":                    "#c2553a",
    "DEX":                        "#3a78c2",
    "Bridge":                     "#c2a83a",
    "Derivatives":                "#7a3ac2",
    "Yield":                      "#3ac28a",
    "Stablecoin":                 "#c23a8a",
    "Other":                      "#888888",
    # Basel III Level-1 event types
    "EF":                         "#3a78c2",   # External Fraud
    "IF":                         "#c2553a",   # Internal Fraud
    "CPBP":                       "#7a3ac2",   # Clients/Products/Business Practices
    "EPWS":                       "#aaaaaa",   # Employment Practices & Workplace Safety (empty)
    "EDPM":                       "#3ac28a",   # Execution, Delivery & Process Mgmt
    "BDSF":                       "#c2a83a",   # Business Disruption & System Failures
    "DPA":                        "#888888",   # Damage to Physical Assets (empty)
    # Chang (SOA) categories
    "SC-Technical":               "#3a78c2",
    "SC-Economic":                "#7a3ac2",
    "Cyber-Operational":          "#c23a8a",
    "Blockchain-Infrastructure":  "#c2a83a",
}


def plot_mean_excess(losses: dict[str, np.ndarray], fp: Path,
                     axis: str = "sector"):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, x in losses.items():
        if len(x) < 30:
            continue
        us, es = mean_excess(x)
        ax.plot(us / 1e6, es / 1e6, label=f"{label}  (n={len(x)})",
                color=PALETTE.get(label, "gray"), lw=1.8, alpha=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Threshold u  (USD m, log)")
    ax.set_ylabel("Mean excess e(u) = E[X-u | X>u]  (USD m, log)")
    ax.set_title(f"Mean excess function — DeFi exploit losses by {axis}\n"
                 "A linearly increasing tail indicates GPD-type heavy tail (ξ>0)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(fp, dpi=140); plt.close(fig)


def plot_hill(losses: dict[str, np.ndarray], fp: Path,
              axis: str = "sector"):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, x in losses.items():
        if len(x) < 30:
            continue
        ks, xis = hill_estimator(x)
        ax.plot(ks, xis, label=f"{label}  (n={len(x)})",
                color=PALETTE.get(label, "gray"), lw=1.6, alpha=0.9)
    ax.axhline(0, color="gray", lw=0.8, alpha=0.5)
    ax.axhline(1, color="red", lw=0.8, ls="--", alpha=0.7,
               label="ξ = 1 (infinite mean threshold)")
    ax.axhline(0.5, color="orange", lw=0.8, ls="--", alpha=0.7,
               label="ξ = 0.5 (infinite variance threshold)")
    ax.set_xlabel("k  (number of upper order statistics)")
    ax.set_ylabel("Hill tail index estimate  ξ̂(k)")
    ax.set_title(f"Hill plot — tail index of DeFi exploit losses by {axis}\n"
                 "Plateau region gives ξ̂; ξ > 1 ⇒ infinite mean, ξ > 0.5 ⇒ infinite variance")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(fp, dpi=140); plt.close(fig)


def plot_gpd_qq(x: np.ndarray, xi: float, beta: float, threshold: float,
                fp: Path, title: str):
    excess = np.sort(x[x > threshold] - threshold)
    n = len(excess)
    # Theoretical quantiles
    p = (np.arange(1, n + 1) - 0.5) / n
    if abs(xi) < 1e-8:
        theo = -beta * np.log(1 - p)
    else:
        theo = beta / xi * ((1 - p) ** (-xi) - 1)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(theo / 1e6, excess / 1e6, alpha=0.7, s=22, color="#3a78c2")
    lim = max(theo.max(), excess.max()) / 1e6
    ax.plot([0, lim], [0, lim], "r--", lw=1.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Theoretical GPD quantile  (USD millions, log)")
    ax.set_ylabel("Empirical excess  (USD millions, log)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(fp, dpi=140); plt.close(fig)


def plot_lda_distribution(totals: np.ndarray, observed_recent: float | None,
                          tvl_usd: float, fp: Path):
    """Histogram of simulated annual aggregate losses + VaR/ES lines."""
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(13, 5.5))
    # Linear view: zoom to 95% quantile to show body of distribution;
    # log view: full range with finer bins out to the heavy tail.
    bins_lin = np.linspace(0, np.quantile(totals, 0.95) / 1e6, 60)
    bins_log = np.linspace(0, np.quantile(totals, 0.9995) / 1e6, 120)
    ax_lin.hist(totals / 1e6, bins=bins_lin, color="#3a78c2",
                edgecolor="white", alpha=0.85)
    ax_log.hist(totals / 1e6, bins=bins_log, color="#3a78c2",
                edgecolor="white", alpha=0.85)
    qs = [0.5, 0.95, 0.99, 0.995, 0.999]
    qcolors = {0.5: "gray", 0.95: "#c2a83a", 0.99: "#c2553a",
               0.995: "#7a3ac2", 0.999: "black"}
    qlabels = {0.5: "Median", 0.95: "VaR 95%", 0.99: "VaR 99%",
               0.995: "VaR 99.5%", 0.999: "VaR 99.9%"}
    for q in qs:
        v = np.quantile(totals, q) / 1e6
        for ax in (ax_lin, ax_log):
            ax.axvline(v, color=qcolors[q], lw=1.6, alpha=0.9,
                       label=f"{qlabels[q]}: USD {v:,.0f}m")
    if observed_recent:
        for ax in (ax_lin, ax_log):
            ax.axvline(observed_recent / 1e6, color="red", lw=2.5,
                       ls=":", label=f"Observed 365d: USD {observed_recent/1e6:,.0f}m")
    ax_lin.set_yscale("linear")
    ax_log.set_yscale("log")
    for ax in (ax_lin, ax_log):
        ax.set_xlabel("Annual aggregate gross loss  (USD millions)")
        ax.set_ylabel("Density of simulated years (count)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    ax_lin.set_xlim(0, np.quantile(totals, 0.95) / 1e6)
    ax_log.set_xlim(0, np.quantile(totals, 0.9995) / 1e6)
    ax_lin.set_title("Linear view (zoom to 95th pct.) — typical years")
    ax_log.set_title("Log y, full tail — rare years carry most of the loss")
    fig.suptitle(f"Loss-distribution approach: annual DeFi exploit aggregate "
                 f"(simulated {len(totals):,} years)  ·  Avg TVL ≈ USD {tvl_usd/1e9:.0f}B",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fp, dpi=140); plt.close(fig)


def _thin_frame(ax):
    """Thin the axes frame and ticks to match the panel line widths."""
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2)


# Panels (a) and (b) of the paper's Figure 1 are each printed at
# 0.49\linewidth of an LLNCS column (= 2.354 in), so the canvas is set to
# that exact width: nothing is scaled on the page and the sizes below are
# the sizes that reach print. Springer requires figure lettering of at
# least 6 pt (~2 mm), so the smallest type here (the legend) is 6.5 pt.
# Both panels share one canvas size and one axes rectangle, so the two plot
# frames coincide exactly when the subfigures are set side by side.
BW_FIGSIZE = (2.354, 1.897)
BW_AXRECT = dict(left=0.24, right=0.975, bottom=0.31, top=0.97)
BW_FS_TICK, BW_FS_LABEL, BW_FS_LEGEND = 6.5, 7.5, 6.5


def plot_lossdist_and_ccdf(h: pd.DataFrame, losses: dict[str, np.ndarray],
                           gpd_fits: dict[str, dict], sectors: list[str],
                           fp_a: Path, fp_b: Path):
    """Two standalone black-and-white figures: (a) per-sector violin loss
    distribution -> fp_a, (b) log-log CCDF with the fitted POT-GPD tail
    -> fp_b. Sectors keyed by marker shape (not colour) for greyscale.

    Panel (b) overlays the GPD actually used for the capital numbers: the
    curve is the POT tail estimator
    P(X > x) = (n_u/n) * (1 + xi (x - u) / beta)^(-1/xi), drawn only from
    the fitted threshold u* upward, which is the whole range over which the
    model is claimed to hold.
    Per-sector n is not repeated on either panel; it is a column of the
    per-sector POT-GPD table, and at print size the panels have room only
    for labels that have to be read."""
    sectors_present = [s for s in sectors if (h["sector"] == s).sum() >= 3]

    # (a) violin loss distribution
    figA, axL = plt.subplots(figsize=BW_FIGSIZE)
    figA.subplots_adjust(**BW_AXRECT)
    log_groups = [np.log10(h.loc[h["sector"] == s, "gross"].values)
                  for s in sectors_present]
    parts = axL.violinplot(log_groups, showmeans=False, showmedians=True,
                           widths=0.8)
    for body in parts["bodies"]:
        body.set_facecolor("0.8"); body.set_edgecolor("black")
        body.set_linewidth(0.5); body.set_alpha(0.7)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("black"); parts[key].set_linewidth(0.5)
    parts["cmedians"].set_linewidth(0.8)
    rng = np.random.default_rng(0)
    for i, x in enumerate(log_groups, 1):
        axL.scatter(i + rng.uniform(-0.15, 0.15, size=len(x)), x,
                    s=1.5, color="black", alpha=0.25)
    axL.set_xticks(range(1, len(sectors_present) + 1))
    axL.set_xticklabels(sectors_present, fontsize=BW_FS_TICK,
                        rotation=45, ha="right", rotation_mode="anchor")
    # Decade ticks labelled in USD rather than as log10 exponents: same
    # scale, but the reader does not have to convert.
    decades = [(3, "1k"), (4, "10k"), (5, "100k"), (6, "1m"),
               (7, "10m"), (8, "100m"), (9, "1B")]
    axL.set_yticks([lvl for lvl, _ in decades])
    axL.set_yticklabels([lab for _, lab in decades], fontsize=BW_FS_TICK)
    axL.set_ylabel("Gross loss  (USD, log)", fontsize=BW_FS_LABEL)
    axL.set_xlim(0.4, len(sectors_present) + 0.6)
    axL.grid(True, axis="y", alpha=0.3, lw=0.4)
    _thin_frame(axL)
    figA.savefig(fp_a); plt.close(figA)

    # (b) log-log CCDF with fitted GPD tail
    figB, axR = plt.subplots(figsize=BW_FIGSIZE)
    figB.subplots_adjust(**BW_AXRECT)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for i, s in enumerate(sectors_present):
        x = losses.get(s)
        if x is None or len(x) < 30:
            continue
        xs = np.sort(x); n = len(xs)
        ccdf = 1 - np.arange(1, n + 1) / (n + 1)
        axR.plot(xs, ccdf, marker=markers[i % len(markers)], ms=2.0, lw=0,
                 markerfacecolor="none", markeredgecolor="black",
                 markeredgewidth=0.35, alpha=0.8, label=s)
        fit = gpd_fits.get(s)
        if fit and np.isfinite(fit.get("xi", np.nan)):
            u, xi, beta = (fit["threshold_usd"], fit["xi"], fit["beta"])
            # Empirical exceedance probability at u anchors the tail curve to
            # the plotted CCDF; (xs > u) matches the strict POT convention.
            p_above = (xs > u).mean()
            if p_above > 0 and beta > 0 and xs.max() > u:
                xx = np.geomspace(u, xs.max(), 200)
                yy = p_above * (1.0 + xi * (xx - u) / beta) ** (-1.0 / xi)
                axR.plot(xx, yy, color="black", ls="--", lw=0.5, alpha=0.6)
    axR.set_xscale("log"); axR.set_yscale("log")
    # labelpad centres the label in the bottom margin, which is sized for
    # panel (a)'s rotated sector labels so both panels share one axes rect.
    axR.set_xlabel("Loss  (USD, log)", fontsize=BW_FS_LABEL, labelpad=12)
    axR.set_ylabel(r"Empirical CCDF  $P(X \geq L)$", fontsize=BW_FS_LABEL)
    axR.tick_params(labelsize=BW_FS_TICK)
    _thin_frame(axR)
    axR.grid(True, alpha=0.3, which="both")
    leg = axR.legend(fontsize=BW_FS_LEGEND, loc="lower left", labelspacing=0.3,
                     handletextpad=0.4, borderpad=0.4, framealpha=0.9)
    leg.get_frame().set_linewidth(0.4)
    figB.savefig(fp_b); plt.close(figB)


def plot_gpd_qq_grid(losses: dict[str, np.ndarray], fits: dict, sectors: list[str],
                     fp: Path):
    """All-sector POT-GPD Q-Q small multiples (2x4 grid; 8th panel blank).
    Each panel plots empirical excesses against GPD theoretical quantiles at
    the plateau threshold, log-log, with the y=x reference line dashed."""
    order = [s for s in sectors
             if s in fits and np.isfinite(fits[s]["pot_gpd"]["xi"])]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6))
    axf = axes.flatten()
    for i, sect in enumerate(order):
        ax = axf[i]
        pot = fits[sect]["pot_gpd"]
        xi = float(pot["xi"]); beta = float(pot["beta"])
        thr = float(pot["threshold_usd"])
        x = np.asarray(losses[sect], dtype=float)
        excess = np.sort(x[x > thr] - thr)
        n = len(excess)
        p = (np.arange(1, n + 1) - 0.5) / n
        if abs(xi) < 1e-8:
            theo = -beta * np.log(1 - p)
        else:
            theo = beta / xi * ((1 - p) ** (-xi) - 1)
        ax.scatter(theo / 1e6, excess / 1e6, s=13, facecolor="none",
                   edgecolor="black", linewidths=0.6, alpha=0.8)
        lim = max(theo.max(), excess.max()) / 1e6
        lo = max(min(theo.min(), excess.min()) / 1e6, 1e-3)
        ax.plot([lo, lim], [lo, lim], color="black", ls="--", lw=1.0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{sect}  ($\\hat\\xi={xi:+.2f}$, "
                     f"$n_u={pot['n_exceedances']}$)", fontsize=10)
        ax.grid(True, alpha=0.3, which="both")
    for j in range(len(order), len(axf)):
        axf[j].axis("off")
    fig.supxlabel("GPD theoretical quantile  (USD m, log)", fontsize=10)
    fig.supylabel("Empirical excess  (USD m, log)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fp, dpi=150); plt.close(fig)


def plot_events_scatter(h_defi: pd.DataFrame, fp: Path):
    """Scatter of every DeFi-protocol event in the consolidated
    dataset, gross USD loss vs date, log-scale on y, colour-coded by
    Basel III L1 category. The caller is responsible for filtering
    to DeFi-protocol events (non-DeFi CEX / wallet / custodian
    records dropped upstream)."""
    h = h_defi.copy().sort_values("date")
    basel_order = ["EF", "IF", "CPBP", "EDPM", "BDSF", "EPWS", "DPA"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for cat in basel_order:
        m = h["basel2_category"] == cat
        if not m.any():
            continue
        ax.scatter(h.loc[m, "date"], h.loc[m, "gross"],
                   s=16, color=PALETTE.get(cat, "gray"),
                   edgecolor="white", linewidth=0.3, alpha=0.75,
                   label=f"{cat} (n={int(m.sum())})")

    ax.set_yscale("log")
    ax.set_ylim(1e3, 2e9)
    ax.set_xlabel("Date")
    ax.set_ylabel("Gross loss  (USD, log)")
    ax.set_title("DeFi-protocol operational-risk events "
                 "— gross loss vs date, by Basel L1 category")
    ax.legend(fontsize=8, loc="lower right", ncol=3, framealpha=0.95)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fp, dpi=140); plt.close(fig)


def plot_loss_distribution_by_sector(h: pd.DataFrame, fp: Path,
                                     sectors: list[str]):
    """Violin/strip plot of log10(gross loss) by sector. Sector order is
    supplied by the caller (the paper's canonical median-descending order)
    so every per-sector figure shares the same axis."""
    sectors_present = [s for s in sectors
                       if (h["sector"] == s).sum() >= 3]
    fig, ax = plt.subplots(figsize=(11, 6))
    log_groups = []
    n_per_sector = []
    for s in sectors_present:
        x = np.log10(h.loc[h["sector"] == s, "gross"].values)
        log_groups.append(x)
        n_per_sector.append(len(x))
    parts = ax.violinplot(log_groups, showmeans=False, showmedians=True,
                          widths=0.8)
    for body in parts["bodies"]:
        # Black-and-white figure: uniform light-grey fill, black outline.
        body.set_facecolor("0.8")
        body.set_edgecolor("black")
        body.set_alpha(0.7)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("black")
    parts["cmedians"].set_linewidth(1.5)
    # Overlay strip plot for sample-size context
    rng = np.random.default_rng(0)
    for i, x in enumerate(log_groups, 1):
        ax.scatter(i + rng.uniform(-0.15, 0.15, size=len(x)), x,
                   s=4, color="black", alpha=0.25)
    ax.set_xticks(range(1, len(sectors_present) + 1))
    ax.set_xticklabels([f"{s}\n(n={n})"
                        for s, n in zip(sectors_present, n_per_sector)],
                       fontsize=9)
    ax.set_ylabel("Gross loss  (log10 USD)")
    # Reference lines for human-readable USD bands
    for lvl, lab in [(3, "USD 1k"), (4, "USD 10k"), (5, "USD 100k"),
                     (6, "USD 1m"), (7, "USD 10m"), (8, "USD 100m"),
                     (9, "USD 1B")]:
        ax.axhline(lvl, color="gray", lw=0.5, alpha=0.4)
        ax.text(len(sectors_present) + 0.55, lvl, lab, va="center",
                fontsize=7, color="gray")
    ax.set_title("Loss distribution by sector "
                 "— per-event gross USD\n"
                 "(violin = density; black bar = median; dots = events)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fp, dpi=140); plt.close(fig)


def plot_rolling_intensity(h: pd.DataFrame, fp: Path, group_col: str,
                            order: list[str], window_days: int = 90,
                            title_suffix: str = "",
                            include_total: bool = False,
                            figsize: tuple[float, float] = (8.5, 5.0)):
    """Rolling event-count intensity per group, day-resolution, with
    a configurable window. Each group is plotted as a colored line on
    a shared axis. Total intensity is omitted by default (use
    include_total=True to overlay it as a black line)."""
    h = h.sort_values("date").copy()
    start = h["date"].min().normalize()
    end = h["date"].max().normalize()
    idx = pd.date_range(start, end, freq="D")
    fig, ax = plt.subplots(figsize=figsize)
    if include_total:
        tot = h.set_index("date").assign(_one=1)["_one"].groupby(level=0).sum()
        tot = tot.reindex(idx, fill_value=0).rolling(window_days, min_periods=1).sum()
        tot_per_unit = tot / window_days * 30
        ax.plot(idx, tot_per_unit, color="black", lw=1.6, alpha=0.9,
                label=f"Total (n={len(h)})")
    for g in order:
        m = h[group_col] == g
        if m.sum() < 3:
            continue
        s = h.loc[m].set_index("date").assign(_one=1)["_one"]
        s = s.groupby(level=0).sum().reindex(idx, fill_value=0)
        per_unit = s.rolling(window_days, min_periods=1).sum() / window_days * 30
        ax.plot(idx, per_unit, color=PALETTE.get(g, "gray"),
                lw=1.3, alpha=0.85, label=f"{g} (n={int(m.sum())})")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Rolling event intensity  (events / 30d, {window_days}-day window)")
    ax.set_title(f"Rolling event intensity{title_suffix}")
    ax.legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fp, dpi=140); plt.close(fig)


def plot_monthly_counts(counts: pd.Series, nb_fit: dict, fp: Path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    ax = axes[0]
    ax.bar(counts.index, counts.values, width=20, color="#3a78c2",
           edgecolor="white", alpha=0.85, label="Monthly hack count")
    ax.axhline(counts.mean(), color="#c2553a", lw=1.5,
               label=f"Mean = {counts.mean():.2f}/month")
    ax.set_xlabel("Month")
    ax.set_ylabel("# DeFi-Protocol hacks")
    ax.set_title("Monthly hack frequency  (in-scope DeFi-Protocol, all chains, since Jan 2021)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    bins = np.arange(0, counts.max() + 2) - 0.5
    counts_h, edges = np.histogram(counts.values, bins=bins)
    ax.bar(edges[:-1] + 0.5, counts_h, width=0.9, color="#3a78c2",
           edgecolor="white", alpha=0.85, label="Observed monthly counts")
    # Poisson PMF
    k = np.arange(int(counts.max()) + 1)
    poi = stats.poisson.pmf(k, counts.mean()) * len(counts)
    ax.plot(k, poi, "o-", color="#c2553a", lw=1.5, ms=5,
            label=f"Poisson(λ={counts.mean():.2f}) expected")
    nb_r = 1 / nb_fit["alpha"]; nb_p = nb_r / (nb_r + nb_fit["mu"])
    nb_pmf = stats.nbinom.pmf(k, nb_r, nb_p) * len(counts)
    ax.plot(k, nb_pmf, "s-", color="#7a3ac2", lw=1.5, ms=5,
            label=f"NB(μ={nb_fit['mu']:.2f}, α={nb_fit['alpha']:.3f}) expected")
    ax.set_xlabel("# hacks in month")
    ax.set_ylabel("# months")
    ax.set_title("Monthly count distribution: Poisson vs negative-binomial fit "
                 "(NB > Poisson ⇒ over-dispersion / clustering)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fp, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Figures and machine-readable risk summary
#
# Regenerates the paper's figures and output/risk_summary.json from the same
# working sample as the tables: the consolidated DeFi set (non-DeFi removed
# upstream) with the depositor-facing filter applied. Called from main() so a
# single `python code.py` run produces every artifact the paper cites.
# ---------------------------------------------------------------------------

def regenerate_figures_and_summary():
    print("[1/7] Loading data ...")
    # Figures are drawn on the same working sample as the tables: the DeFi
    # consolidated set (non-DeFi already removed upstream) with the
    # depositor-facing filter applied.
    h = load_hacks()
    h = h.loc[depositor_facing_mask(h)].reset_index(drop=True)
    panel = load_tvl_panel()
    print(f"  in-scope DeFi-Protocol hacks: n = {len(h)}")
    print(f"  date range: {h['date'].min().date()} .. {h['date'].max().date()}")

    sector_candidates = ["Lending", "DEX", "Bridge", "Yield",
                         "Stablecoin", "Derivatives", "Other"]
    sectors = sorted(
        [s for s in sector_candidates if (h["sector"] == s).any()],
        key=lambda s: -h.loc[h["sector"] == s, "gross"].median(),
    )
    basel_cats = ["EF", "IF", "CPBP", "EDPM", "BDSF"]   # EPWS & DPA empty in DeFi

    def _losses_by(col: str, labels: list[str]) -> dict[str, np.ndarray]:
        d = {}
        for lab in labels:
            d[lab] = h.loc[h[col] == lab, "gross"].values
        return d

    losses        = _losses_by("sector",          sectors)
    losses_basel  = _losses_by("basel2_category", basel_cats)

    # ---- 2. Severity tail fits ---------------------------------------------
    print("[2/6] Severity tail fits ...")

    def fit_axis(d: dict[str, np.ndarray]) -> dict[str, dict]:
        out = {}
        for label, x in d.items():
            if len(x) < 20:
                continue
            # Plateau-stability threshold rule (top-down, see
            # select_threshold_plateau docstring): pick the highest q at
            # which xi-hat(q) is stable with respect to xi-hat(q - 0.05),
            # subject to an absolute exceedance-count floor (n_u >= 20,
            # EKM 1997 Sec. 6.4) and a relative one (n_u >= ceil(0.10*n),
            # DuMouchel 1983 "top 10%" rule).
            sel = select_threshold_plateau(x, tau=0.30,
                                            n_u_abs=20, n_u_frac=0.10)
            q0 = sel["q_star"]
            pot = fit_pot_gpd(x, threshold_q=q0)
            # Parametric bootstrap CIs on (xi, beta) at the headline threshold:
            boot = bootstrap_pot_gpd(x, threshold_q=q0, n_boot=1000, rng=RNG)
            scan = {}
            for q in (0.50, 0.60, 0.70, 0.80, 0.90):
                f = fit_pot_gpd(x, threshold_q=q)
                scan[f"{q:.2f}"] = {"xi": f["xi"], "beta": f["beta"],
                                    "n_exc": f["n_exceedances"],
                                    "threshold_usd": f["threshold_usd"]}
            out[label] = {"pot_gpd": pot,
                          "pot_bootstrap": boot,
                          "pot_threshold_scan": scan,
                          "plateau_selection": sel,
                          "n": int(len(x)),
                          "sum_usd": float(x.sum()),
                          "max_usd": float(x.max()),
                          "median_usd": float(np.median(x))}
            if pot.get("xi") is not None and np.isfinite(pot["xi"]):
                print(f"  {label:<26s} n={len(x):>4d}  POT-GPD ξ̂={pot['xi']:+.3f}  "
                      f"[{boot['xi_lo']:+.2f}, {boot['xi_hi']:+.2f}]  "
                      f"β̂=${pot['beta']/1e6:.1f}m  thresh=${pot['threshold_usd']/1e6:.1f}m")
        return out

    print(" -- by sector --")
    fits        = fit_axis(losses)
    print(" -- by Basel II Level-1 (computed for output JSON only) --")
    fits_basel  = fit_axis(losses_basel)

    # ---- 3. Plots: mean excess + Hill + log-log CCDF + GPD Q-Q -------------
    # The paper renders per-sector EVT diagnostics only; per-Basel results
    # are retained in the output JSON but no longer plotted (the paper
    # focuses on sector-level capital requirements).
    print("[3/6] Plotting EVT diagnostics (sector axis only) ...")

    keep_sector = {k: v for k, v in losses.items() if k in fits}
    plot_mean_excess(keep_sector, FIG / "r1_mean_excess.png", axis="sector")
    plot_hill(keep_sector, FIG / "r2_hill_plot.png", axis="sector")
    # Figures 1 and 2 of the paper: loss distribution and CCDF, as two
    # separate black-and-white figures.
    gpd_fits_sector = {k: v["pot_gpd"] for k, v in fits.items()}
    plot_lossdist_and_ccdf(h, keep_sector, gpd_fits_sector, sectors,
                           FIG / "r0b_violin_bw.pdf",
                           FIG / "r3_ccdf_bw.pdf")
    # Per-sector POT-GPD QQ plots — one for Lending (the headline
    # capital sector and diagnostic exemplar).
    pot_lending = fits["Lending"]["pot_gpd"]
    plot_gpd_qq(losses["Lending"], pot_lending["xi"], pot_lending["beta"],
                pot_lending["threshold_usd"], FIG / "r4_gpd_qq_lending.png",
                title=(f"POT-GPD Q-Q  ·  DeFi-Lending  ·  n_exceed="
                       f"{pot_lending['n_exceedances']}\n"
                       f"ξ̂={pot_lending['xi']:.3f}  "
                       f"β̂=USD {pot_lending['beta']/1e6:.1f}m  "
                       f"threshold (q={pot_lending['threshold_q']:.0%}) = "
                       f"USD {pot_lending['threshold_usd']/1e6:.1f}m"))
    for sect in sectors:
        if sect == "Lending" or sect not in fits:
            continue
        pot_s = fits[sect]["pot_gpd"]
        if not np.isfinite(pot_s["xi"]):
            continue
        plot_gpd_qq(losses[sect], pot_s["xi"], pot_s["beta"],
                    pot_s["threshold_usd"],
                    FIG / f"r4_gpd_qq_{sect.lower()}.png",
                    title=(f"POT-GPD Q-Q  ·  DeFi-{sect}  ·  n_exceed="
                           f"{pot_s['n_exceedances']}\n"
                           f"ξ̂={pot_s['xi']:.3f}  "
                           f"β̂=USD {pot_s['beta']/1e6:.1f}m  "
                           f"threshold (q={pot_s['threshold_q']:.0%}) = "
                           f"USD {pot_s['threshold_usd']/1e6:.1f}m"))
    # All-sector POT-GPD Q-Q grid (paper figure; blank 8th panel).
    plot_gpd_qq_grid(losses, fits, sectors, FIG / "r4_gpd_qq_all_sectors.png")
    # Companion threshold-stability diagnostic (xi-hat vs quantile plateau).
    _plot_xi_threshold_stability(keep_sector,
                                 FIG / "r5_xi_threshold_stability.png")

    # ---- 4. Frequency model + monthly counts -------------------------------
    print("[4/6] Frequency model (2020-2026 window) ...")
    counts = monthly_counts(h, start=None)  # post-2020 (h already filtered)
    disp = dispersion_test(counts.values)
    nb = fit_negbin(counts.values)
    pois_ll = (stats.poisson.logpmf(counts.values, counts.values.mean())).sum()
    nb_r = 1 / nb["alpha"]; nb_p = nb_r / (nb_r + nb["mu"])
    nb_ll = stats.nbinom.logpmf(counts.values, nb_r, nb_p).sum()
    lr_stat = 2 * (nb_ll - pois_ll)
    lr_p = 1 - stats.chi2.cdf(lr_stat, df=1)
    print(f"  monthly mean = {disp['mean']:.2f}  var = {disp['var']:.2f}  "
          f"D = var/mean = {disp['D']:.2f}  z-overdisp = {disp['z']:.2f}  "
          f"p = {disp['p_overdisp']:.3g}")
    print(f"  NB(μ={nb['mu']:.2f}, α={nb['alpha']:.3f})  "
          f"vs Poisson LR = {lr_stat:.2f}  p = {lr_p:.3g}")
    plot_monthly_counts(counts, nb, FIG / "r6_monthly_counts.png")

    # Exploratory plots for §4 — DeFi-protocol events 2020-2026.
    # The 2020 cutoff matches Aramonte et al. (2021)'s dating of the
    # DeFi-as-a-non-trivial-financial-substrate era to 2020-Q3, and is
    # applied uniformly across every plot and model fit in the paper.
    # We pass the post-2020-filtered DataFrame `h` (already filtered
    # by load_hacks) to all §4 plot helpers.
    plot_events_scatter(h, FIG / "r0_events_scatter.png")
    plot_loss_distribution_by_sector(h,
                                     FIG / "r0b_loss_distribution_by_sector.png",
                                     sectors=sectors)
    plot_rolling_intensity(h, FIG / "r0c_rolling_intensity_sector.png",
                           group_col="sector", order=sectors,
                           window_days=90,
                           title_suffix=" by sector (90-day window)",
                           include_total=False)
    plot_rolling_intensity(h, FIG / "r0d_rolling_intensity_basel.png",
                           group_col="basel2_category",
                           order=["EF", "IF", "CPBP", "EDPM", "BDSF"],
                           window_days=90,
                           title_suffix=" by event type (90-day window)",
                           include_total=False)

    # Per-sector LDA follows in 5b below.
    years_obs_full = (h["date"].max() - h["date"].min()).days / 365.25
    tvl_recent = float(panel["DeFi"].iloc[-365:].mean())

    # ---- 5. Per-sector LDA: bank-style capital requirements -------------
    # For each sector with a sensible event count and a TVL denominator
    # in the panel, fit a compound-Poisson body+GPD-tail LDA and report
    # mean / VaR99 / ES99 / ES99.9 as bps of the trailing-365d sector-
    # average TVL.
    print("[5/6] Per-sector capital-requirements LDA ...")
    sector_tvl_map = {s: s for s in
                      ("Lending", "DEX", "Bridge", "Yield", "Derivatives",
                       "Stablecoin", "Other")}
    # Iterate in canonical median-descending order so the LDA capital
    # table matches the figure axis ordering.
    # Per-sector monthly NB fits. We model the per-sector annual event
    # count in the LDA as NB(λ_s, α_s) with α_s estimated from each
    # sector's own monthly count series; this avoids the cross-sector
    # pooling and the time-aggregation invariance argument required by
    # the previous version.
    pooled_nb_alpha = float(nb["alpha"])
    sector_nb_alpha: dict[str, float] = {}
    sector_nb_lr: dict[str, dict] = {}   # per-sector NB-vs-Poisson LR test (Table nb)
    for sect in sectors:
        sub = h[h["sector"] == sect]
        if len(sub) < 30:
            sector_nb_alpha[sect] = pooled_nb_alpha
            continue
        full_months = pd.date_range(h["date"].min().strftime("%Y-%m-01"),
                                      h["date"].max(), freq="MS")
        c = (sub.set_index("date").resample("MS").size()
                .reindex(full_months, fill_value=0).values.astype(float))
        try:
            f = fit_negbin(c)
            sector_nb_alpha[sect] = float(f["alpha"])
            # Per-sector NB-vs-Poisson likelihood-ratio test (chi^2, df=1),
            # the "LR vs Poi" / "p" columns of the NB table.
            _r = 1.0 / f["alpha"]; _p = _r / (_r + f["mu"])
            _nb_ll = float(stats.nbinom.logpmf(c, _r, _p).sum())
            _pois_ll = float(stats.poisson.logpmf(c, c.mean()).sum())
            _lr = 2.0 * (_nb_ll - _pois_ll)
            sector_nb_lr[sect] = {
                "mu_monthly": float(f["mu"]),
                "lr_vs_poisson": float(_lr),
                "p_value": float(1 - stats.chi2.cdf(_lr, df=1)),
            }
        except Exception:
            sector_nb_alpha[sect] = pooled_nb_alpha
        print(f"  NB[{sect:<12s}]  α̂_s = {sector_nb_alpha[sect]:.3f}"
              + (f"  LR={sector_nb_lr[sect]['lr_vs_poisson']:.1f}"
                 f"  p={sector_nb_lr[sect]['p_value']:.2g}"
                 if sect in sector_nb_lr else ""))
    _caps_blob = json.loads((DATA / "raw" / "defillama" /
                  "sector_exposure_caps_2026-06-30.json").read_text())
    SECTOR_CAPS = dict(_caps_blob["caps_usd"])
    # Stablecoin exposure base: decentralized (non-fiat-backed) stablecoin
    # circulating supply only. Fiat-backed issuers (USDT, USDC, ...) are
    # centrally issued and generate none of the DeFi-stablecoin operational-
    # risk events, so counting their ~USD 285B supply would inflate the
    # denominator ~12x. Derived from the cached DefiLlama peggedAssets
    # snapshot; the sector cap is the largest single decentralized coin.
    _stables = json.loads((DATA / "raw" / "defillama" /
                  "stablecoins_2026-06-30.json").read_text())
    def _coin_usd(c):
        circ = c.get("circulating") or {}
        amt = list(circ.values())[0] if circ else 0.0
        return float(amt or 0.0) * float(c.get("price") or 1.0)
    _dec = [c for c in _stables
            if (c.get("pegMechanism") or "") != "fiat-backed"]
    STABLE_DENOM_USD = float(sum(_coin_usd(c) for c in _dec))
    _stable_cap_coin = max(_dec, key=_coin_usd)
    SECTOR_CAPS["Stablecoin"] = float(_coin_usd(_stable_cap_coin))
    print(f"  Stablecoin denom (decentralized only) = USD "
          f"{STABLE_DENOM_USD/1e9:.1f}B  cap = {_stable_cap_coin['symbol']} "
          f"(USD {SECTOR_CAPS['Stablecoin']/1e9:.1f}B)")
    # "Other" loss cap: largest single-protocol exposure among the
    # categories that make up the Other denominator, on the same
    # largest-single-protocol basis as every other sector (uniform rule).
    # The Other denominator is the panel residual (DeFi minus the five
    # named sectors), which by the panel's construction is exactly the RWA
    # and Algo-Stables buckets, so the cap is drawn from those same
    # categories (in practice a tokenized-treasury RWA venue). Categories
    # that belong to a named sector (staking/restaking, basis-trading,
    # canonical bridges, ...) are already tracked under those sectors, and
    # non-DeFi categories (CEX, OTC, ...) are outside the panel entirely.
    # Derived from the cached DefiLlama /protocols snapshot.
    _OTHER_CATS = {"RWA", "Algo-Stables", "Reserve Currency"}
    _protocols = json.loads((DATA / "raw" / "defillama" /
                  "protocols_2026-06-30.json").read_text())
    _other_prot = [p for p in _protocols
                   if (p.get("category") or "") in _OTHER_CATS
                   and isinstance(p.get("tvl"), (int, float))]
    _other_cap_prot = max(_other_prot, key=lambda p: p["tvl"])
    SECTOR_CAPS["Other"] = float(_other_cap_prot["tvl"])
    print(f"  Other cap = {_other_cap_prot['name']} "
          f"({_other_cap_prot.get('category')}) "
          f"USD {SECTOR_CAPS['Other']/1e9:.1f}B")
    # "Other" (residual) exposure base: total DeFi TVL minus the five
    # TVL-panel sectors, trailing-365d mean (approximate).
    _named = ["Lending", "DEX", "Bridge", "Derivatives", "Yield"]
    OTHER_DENOM_USD = float(
        (panel["DeFi"] - panel[_named].sum(axis=1)).iloc[-365:].mean())
    sector_lda = {}
    for sect in sectors:
        if sect not in sector_tvl_map:
            continue
        x_s = losses.get(sect)
        if x_s is None or len(x_s) < 30:
            continue
        pot_s = fits[sect]["pot_gpd"]
        years_obs = (h["date"].max() - h["date"].min()).days / 365.25
        n_s = (h["sector"] == sect).sum()
        lam_s = n_s / years_obs
        if sect == "Stablecoin":
            # No supply-side TVL; use trailing-365d mean stablecoin
            # circulating supply as the exposure denominator (approximate).
            tvl_s = STABLE_DENOM_USD
        elif sect == "Other":
            # Residual: total DeFi TVL minus the named sectors (approximate).
            tvl_s = OTHER_DENOM_USD
        else:
            tvl_col = sector_tvl_map[sect]
            tvl_s = float(panel[tvl_col].iloc[-365:].mean())
        cap_s = float(SECTOR_CAPS.get(sect, float("inf")))
        samp_s = severity_sampler_pot(x_s, pot_s["threshold_usd"],
                                       pot_s["xi"], pot_s["beta"], cap=cap_s)

        # Headline LDA uses NB frequency with per-sector α_s
        # estimated from each sector's monthly count series.
        alpha_s = sector_nb_alpha.get(sect, pooled_nb_alpha)
        alpha_annual = alpha_s / 12.0   # i.i.d.-monthly -> annual NB aggregation
        _rng_state = RNG.bit_generator.state   # pair the uncapped draw below
        tot_s = lda_simulate(lam_s, samp_s, n_years=200_000, rng=RNG,
                              frequency="nb", nb_alpha=alpha_annual)
        qs = {q: float(np.quantile(tot_s, q))
              for q in (0.5, 0.95, 0.99, 0.995, 0.999)}
        mean_s = float(tot_s.mean())
        var999_s = qs[0.999]
        var99_s  = qs[0.99]
        var995_s = qs[0.995]

        # Cap-sensitivity diagnostic: the same simulation with NO exposure cap.
        # The global RNG advanced only through the capped sim above, so this
        # leaves every downstream result unchanged; reseeding a fresh generator
        # from the captured state pairs each uncapped year to its capped
        # counterpart (capped loss = min(uncapped, cap) event-wise). A sector
        # whose cap never binds at the 99.9% level returns an identical figure;
        # a cap-bound sector reveals how much of the reported VaR is the cap.
        _rng_pair = np.random.default_rng()
        _rng_pair.bit_generator.state = _rng_state
        samp_uncap = severity_sampler_pot(x_s, pot_s["threshold_usd"],
                                          pot_s["xi"], pot_s["beta"], cap=None)
        tot_uncap = lda_simulate(lam_s, samp_uncap, n_years=200_000, rng=_rng_pair,
                                 frequency="nb", nb_alpha=alpha_annual)
        var999_uncapped = float(np.quantile(tot_uncap, 0.999))

        # Bootstrap CIs on LDA bps (NB)
        boot_lda = bootstrap_lda_bps(
            x_s, pot_s["threshold_q"], lam_s, tvl_s,
            n_boot=200, n_years=50_000,
            frequency="nb", nb_alpha=alpha_annual, cap=cap_s, rng=RNG)

        sector_lda[sect] = {
            "n_events": int(n_s),
            "lambda_yr": float(lam_s),
            "nb_alpha_monthly": float(alpha_s),
            "nb_alpha_applied": float(alpha_annual),
            "exposure_cap_usd": cap_s,
            "tvl_recent_usd": tvl_s,
            "xi": float(pot_s["xi"]),
            "beta_usd": float(pot_s["beta"]),
            "threshold_usd": float(pot_s["threshold_usd"]),
            "mean_loss_usd":     mean_s,
            "var999_usd":        var999_s,
            "mean_loss_bps":     mean_s   / tvl_s * 1e4,
            "var999_bps":        var999_s / tvl_s * 1e4,
            "var999_uncapped_usd": var999_uncapped,
            "var999_uncapped_bps": var999_uncapped / tvl_s * 1e4,
            "var99_usd":         var99_s,
            "var995_usd":        var995_s,
            "var99_bps":         var99_s  / tvl_s * 1e4,
            "var995_bps":        var995_s / tvl_s * 1e4,
            "bootstrap":         boot_lda,
        }
        print(f"  {sect:<13s} n={n_s:>3d}  λ={lam_s:5.1f}/yr (NB α_yr={alpha_annual:.3f})  "
              f"TVL=USD {tvl_s/1e9:5.1f}B  cap=USD {cap_s/1e9:4.1f}B  "
              f"VaR99.9={var999_s/1e6:>7.0f}m ({var999_s/tvl_s*1e4:>5.0f}bps "
              f"[{boot_lda['var999_bps_lo']:>5.0f}, {boot_lda['var999_bps_hi']:>5.0f}])")

    # ---- 6. Top-10 Lending protocol adequacy ----------------------------
    print("[6/6] Top-10 Lending-protocol capital adequacy ...")
    _lend = sector_lda["Lending"]
    lending_sector_fit = {
        "losses":        losses["Lending"],
        "threshold_usd": _lend["threshold_usd"],
        "xi":            _lend["xi"],
        "beta":          _lend["beta_usd"],
        "lam_s":         _lend["lambda_yr"],
        "alpha_annual":  _lend["nb_alpha_applied"],
        "tvl_s":         _lend["tvl_recent_usd"],
    }
    top10 = top_n_lending_adequacy(
        n=10, sector_fit=lending_sector_fit,
        gross_tvl_path=DATA / "raw" / "defillama" / "lending_gross_tvl_2026-06-30.json",
        rng=np.random.default_rng(20260701),
    )
    agg_tvl     = sum(p["tvl_usd"]    for p in top10)
    agg_var999  = sum(p["var999_usd"] for p in top10)
    agg_var99   = sum(p["var99_usd"]  for p in top10)
    agg_var995  = sum(p["var995_usd"] for p in top10)
    agg_buffer  = sum(p["buffer_usd"] for p in top10)
    agg_gap     = sum(p["gap_usd"]    for p in top10)
    # Buffered-only aggregates (the 4 venues that disclose a reserve)
    _buf = [p for p in top10 if p["buffer_usd"] > 0]
    bufcov = {
        "n_buffered":     len(_buf),
        "cov99_pct":   100 * sum(p["buffer_usd"] for p in _buf) / sum(p["var99_usd"]  for p in _buf),
        "cov995_pct":  100 * sum(p["buffer_usd"] for p in _buf) / sum(p["var995_usd"] for p in _buf),
        "cov999_pct":  100 * sum(p["buffer_usd"] for p in _buf) / sum(p["var999_usd"] for p in _buf),
        "mean_cov99_pct":  100 * float(np.mean([p["coverage_99"]  for p in _buf])),
        "mean_cov995_pct": 100 * float(np.mean([p["coverage_995"] for p in _buf])),
        "mean_cov999_pct": 100 * float(np.mean([p["coverage"]     for p in _buf])),
    }
    print(f"  buffered-venue coverage: VaR99={bufcov['mean_cov99_pct']:.1f}%  "
          f"VaR99.5={bufcov['mean_cov995_pct']:.1f}%  "
          f"VaR99.9={bufcov['mean_cov999_pct']:.1f}%  (mean of 4)")
    print(f"  Lending sector VaR (%TVL): 99={_lend['var99_bps']/100:.1f}  "
          f"99.5={_lend['var995_bps']/100:.1f}  99.9={_lend['var999_bps']/100:.1f}")
    for p in top10:
        cov = (p["coverage"] * 100) if p["coverage"] == p["coverage"] else float("nan")
        print(f"  {p['name']:<18s}  TVL=USD {p['tvl_usd']/1e9:5.2f}B  "
              f"VaR99.9=USD {p['var999_usd']/1e9:5.2f}B  "
              f"buffer=USD {p['buffer_usd']/1e9:5.2f}B  "
              f"cov={cov:5.1f}%  gap=USD {p['gap_usd']/1e9:5.2f}B")
    print(f"  AGG: TVL={agg_tvl/1e9:.1f}B  VaR99.9={agg_var999/1e9:.2f}B  "
          f"buffers={agg_buffer/1e9:.2f}B  gap={agg_gap/1e9:.2f}B  "
          f"({agg_buffer/agg_var999*100:.0f}% covered)")

    # ---- 6c. Yield-side compensation (S3-r2) -----------------------------
    # Closes the "ex-ante yield pricing" alternative interpretation
    # (perspective-reviewer W3): if DeFi LPs are not capitalised on the
    # protocol side, are they at least compensated through higher yields?
    print("[6c] Yield-side compensation analysis (DefiLlama pools API) ...")
    try:
        yields_df = load_defillama_yields()
        pure_premium_bps = {s: r["mean_loss_bps"]
                              for s, r in sector_lda.items()}
        yield_comp = yield_compensation_table(
            yields_df, sector_pure_premium_bps=pure_premium_bps,
            tbill_pct=3.70)
        for sect, blk in yield_comp["by_sector"].items():
            pp_pct = blk["pure_premium_pct"]
            base = blk["tvl_weighted_base_pct"]
            total = blk["tvl_weighted_total_pct"]
            print(f"  {sect:<10s}  pure_premium={pp_pct:.2f}%  "
                  f"stable-supply base APY={base:.2f}%  "
                  f"total APY={total:.2f}%  "
                  f"excess vs T-bill (base)={base - 3.70:+.2f}%  "
                  f"(total)={total - 3.70:+.2f}%")
    except FileNotFoundError:
        print("  yields_pools.json missing; run the DefiLlama yields fetch")
        yield_comp = None

    # ---- 6d. Comparing the two responses (buffer vs yield) ---------------
    # Response 1 (protocol buffer, step 6) vs Response 2 (depositor yield
    # spread, step 6c): do they substitute? (Section "Comparing the two
    # responses".)
    response_comparison = None
    if yield_comp is not None:
        response_comparison = compare_responses(
            top10,
            yield_comp["by_sector"]["Lending"]["protocols"],
            tbill_pct=3.70,
        )
        rc = response_comparison
        if rc.get("n", 0) >= 3:
            print("[6d] Comparing the two responses (top-10 Lending) ...")
            print(f"  risk spread: {rc['risk_spread_min_bps']:+.0f} to "
                  f"{rc['risk_spread_max_bps']:+.0f} bps "
                  f"(range {rc['risk_spread_range_bps']:.0f}, "
                  f"std {rc['risk_spread_std_bps']:.0f})")
            print(f"  mean spread: zero-buffer "
                  f"{rc['mean_spread_zero_buffer_bps']:+.0f} bps "
                  f"(n={rc['n_zero_buffer']})  vs  buffered "
                  f"{rc['mean_spread_buffered_bps']:+.0f} bps "
                  f"(n={rc['n_buffered']}, mean cov "
                  f"{rc['mean_coverage_buffered']*100:.0f}%)")

    # ---- 6b. Sensitivity re-fits (R6 + R8) -------------------------------
    print("[6b] Sensitivity re-fits (Bridge w/o Kelp, Derivatives split, "
          "Lending blue-chip subset) ...")
    sensitivity = sensitivity_fits(h, fits, rng=RNG)
    for k, v in sensitivity.items():
        if not isinstance(v, dict):
            continue
        ci = v.get("xi_ci", [float("nan"), float("nan")])
        print(f"  {k:<32s} n={v['n']:>3d}  ξ̂={v['xi']:+.2f}  "
              f"[{ci[0]:+.2f}, {ci[1]:+.2f}]  u=USD {v['threshold_usd']/1e6:.1f}m")

    years_obs = (h["date"].max() - h["date"].min()).days / 365.25

    # ---- Persist summary --------------------------------------------------
    print("Saving risk summary ...")
    summary = {
        "as_of": str(h["date"].max().date()),
        "n_hacks_in_scope": int(len(h)),
        "years_observed": float(years_obs),
        "data_source": "data/events_consolidated.csv "
                       "(DefiLlama + rekt.news + kismp123 + DeFiHackLabs + "
                       "BlockSec + de.fi/rekt-database + SlowMist Hacked; "
                       "see events_consolidation.py for source-precedence and "
                       "deduplication methodology)",
        "fits":       fits,
        "fits_basel": fits_basel,
        "frequency": {
            "fit_window": f"{h['date'].min().date()} .. {h['date'].max().date()}",
            "monthly_counts": {
                "mean": disp["mean"], "var": disp["var"], "D": disp["D"],
                "p_overdispersion": disp["p_overdisp"],
                "n_months": int(len(counts))},
            "neg_binomial": nb,
            "lr_nb_vs_poisson": {"stat": float(lr_stat), "p": float(lr_p)},
            "per_sector_alpha": sector_nb_alpha,
            "per_sector_lr_vs_poisson": sector_nb_lr,
        },
        "lda": {
            "tvl_recent_usd_defi_total": tvl_recent,
            "per_sector": sector_lda,
        },
        "protocol_adequacy_top10_lending": {
            "sector_var999_bps": float(sector_lda["Lending"]["var999_bps"]),
            "protocols": top10,
            "aggregate_tvl_usd":    float(agg_tvl),
            "aggregate_var999_usd": float(agg_var999),
            "aggregate_buffer_usd": float(agg_buffer),
            "aggregate_gap_usd":    float(agg_gap),
            "aggregate_coverage":   float(agg_buffer / agg_var999) if agg_var999 else 0.0,
            "quantile_robustness":  bufcov,
            "sector_var99_bps":     float(sector_lda["Lending"]["var99_bps"]),
            "sector_var995_bps":    float(sector_lda["Lending"]["var995_bps"]),
        },
        "sensitivity_fits": sensitivity,
        "yield_compensation": yield_comp,
        "response_comparison": response_comparison,
    }
    (OUT / "risk_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote risk summary to: {OUT / 'risk_summary.json'}")
    print(f"Wrote new figures to:  {FIG}")


# ---------------------------------------------------------------------------
# 9. Camera-ready pipeline: sector-mistag correction + depositor-facing
#    filter + per-sector LDA + Tables 3/4/5/6/7/8 + §5 robustness.
#
# This is the default entry point (see main() at the bottom). The reference
# fitting functions above (§§1-4) are used unchanged; this section applies
# the two audit filters introduced in the camera-ready revision, refits per
# sector on the filtered sample, and prints every table the paper cites.
# ---------------------------------------------------------------------------

import re                                      # noqa: E402
from collections import defaultdict            # noqa: E402
from scipy.stats import genpareto              # noqa: E402


def bootstrap_lda_bps_parametric(x: np.ndarray, threshold_q: float,
                                  annual_lambda: float, tvl: float,
                                  n_boot: int = 200, n_years: int = 50_000,
                                  frequency: str = "poisson",
                                  nb_alpha: float | None = None,
                                  cap: float | None = None,
                                  rng=None) -> dict:
    """Full parametric-bootstrap analogue of bootstrap_lda_bps.

    For each of `n_boot` replicates: simulate n events from the fitted
    body+GPD mixture (empirical body for X <= u*, GPD(xi_hat, beta_hat)
    shifted by u* for X > u*); refit POT-GPD on the synthetic sample at the
    same q*; build the (capped) mixture severity sampler and Monte-Carlo the
    compound annual aggregate under the original frequency; record mean-loss
    and VaR99.9 in bps of TVL.

    Frequency parameters are held fixed, mirroring the non-parametric variant.
    Isolates severity-fit uncertainty and stays on the same distributional
    footing as the parametric CI on xi_hat in bootstrap_pot_gpd (Table 4).
    """
    if rng is None:
        rng = np.random.default_rng(456)
    n = len(x)
    u = float(np.quantile(x, threshold_q))
    below = x[x <= u]
    p_below = len(below) / n
    exceed = x[x > u] - u
    xi_hat, beta_hat, _ = gpd_mle(exceed)
    if not (np.isfinite(xi_hat) and np.isfinite(beta_hat) and beta_hat > 0):
        return {"method": "parametric", "n_bootstrap": 0,
                "var999_bps_iqr_lo": float("nan"),
                "var999_bps_iqr_hi": float("nan"),
                "var999_bps_lo": float("nan"),
                "var999_bps_hi": float("nan")}
    rows = []
    for _ in range(n_boot):
        u_draw = rng.random(n)
        is_tail = u_draw >= p_below
        n_tail = int(is_tail.sum())
        n_body = n - n_tail
        sim = np.empty(n)
        if n_body > 0:
            sim[~is_tail] = rng.choice(below, size=n_body, replace=True)
        if n_tail > 0:
            sim[is_tail] = u + genpareto.rvs(
                c=xi_hat, loc=0, scale=beta_hat,
                size=n_tail, random_state=rng,
            )
        u_b = float(np.quantile(sim, threshold_q))
        excess_b = sim[sim > u_b] - u_b
        if len(excess_b) < 10:
            continue
        xi_b, beta_b, _ = gpd_mle(excess_b)
        if not (np.isfinite(xi_b) and np.isfinite(beta_b) and beta_b > 0):
            continue
        samp = severity_sampler_pot(sim, u_b, xi_b, beta_b, cap=cap)
        tot = lda_simulate(annual_lambda, samp, n_years=n_years, rng=rng,
                            frequency=frequency, nb_alpha=nb_alpha)
        rows.append((tot.mean() / tvl * 1e4,
                      float(np.quantile(tot, 0.999)) / tvl * 1e4))
    if len(rows) < 30:
        return {"method": "parametric", "n_bootstrap": len(rows),
                "var999_bps_iqr_lo": float("nan"),
                "var999_bps_iqr_hi": float("nan"),
                "var999_bps_lo": float("nan"),
                "var999_bps_hi": float("nan")}
    arr = np.array(rows)
    return {
        "method": "parametric",
        "n_bootstrap": len(rows),
        "mean_bps_lo": float(np.quantile(arr[:, 0], 0.025)),
        "mean_bps_hi": float(np.quantile(arr[:, 0], 0.975)),
        "var999_bps_lo": float(np.quantile(arr[:, 1], 0.025)),
        "var999_bps_hi": float(np.quantile(arr[:, 1], 0.975)),
        "var999_bps_iqr_lo": float(np.quantile(arr[:, 1], 0.25)),
        "var999_bps_iqr_hi": float(np.quantile(arr[:, 1], 0.75)),
        "var999_bps_median": float(np.median(arr[:, 1])),
    }


# --- 9a. Sector re-audit + non-DeFi removal --------------------------------
# The sector mistag correction and NOT_DEFI removal now run upstream in
# events_consolidation.py (see build_reassignment_map / apply_sector_reassignment
# there). events_consolidated.csv already contains only sector-corrected DeFi
# events, so the camera-ready loader below applies only the depositor-facing
# filter.


# --- 9b. Depositor-facing filter (rules R1..R6) ----------------------------

_DEPOSITOR_FILTER_RULES = {
    "R1_gov_token_dilution": re.compile(
        r"(?:governance[- ]?token|utility[- ]?token|reward[- ]?token)|"
        r"token[- ]?(?:dilut|holder(?:s)?[- ]?(?:dilut|loss))|"
        r"(?:unauthorized|infinite)[- ]?mint(?:ing|ed)?|"
        r"(?:mint(?:ing|ed)?|dumped)\s+[\d,]*\s*\$?[A-Z][A-Z0-9]{1,10}\s+tokens?|"
        r"\$(?:COMP|COVER|FAVOR|FNX|PAID|GALA|SDOG|RPL|GB|SPELL|CRV|FFF|"
        r"TICKER|BUILD|CASPER|DAFI|RAI|TRUSTA|POLYNOMIAL)\b|"
        r"aBNBc.*(?:mint|infinite)|insider[- ]?trad(?:ing|e)|"
        r"token[- ]?holder\s+(?:dilut|loss)|over[- ]?distribut",
        re.I),
    "R2_over_liquidation": re.compile(
        r"over[- ]?liquidat|suppliers?\s+(?:were\s+)?(?:made\s+)?whole|"
        r"oracle spike.*(?:liquidat|borrower)|liquidator.*profit|"
        r"DAI.*Coinbase oracle|price spike.*liquidat",
        re.I),
    "R3_treasury_drain": re.compile(
        r"treasur(?:y|ies)\s*(?:breach|drain|hack|steal|hit|reentran)|"
        r"(?:DAO|foundation|team|project|operational)[- ]?controlled|"
        r"(?:DAO|foundation|team|operational)\s+wallet(?:s)?\s+"
        r"(?:drain|hack|breach|steal)|"
        r"protocol treasury\s+(?:drain|breach|hit)|"
        r"company funds(?:\s+used|\s+stolen)|"
        r"portfolio[- ]?tracker.{0,20}treasury",
        re.I),
    "R4_router_approval_hijack": re.compile(
        r"router[- ]?approval[- ]?(?:hijack|drain|exploit)|"
        r"aggregator.*approval[- ]?(?:hijack|drain|exploit)|"
        r"pre[- ]?approvals?\s+.{0,20}(?:drain|steal|hijack)|"
        r"transferFrom.*(?:arbitrary|drain|malicious)|permit2 route|"
        r"RouteProcessor(?:2)?\s*(?:proc|drain|exploit)|"
        r"(?:TransitSwap|Transit\s*Finance|LiFi|GasZipFacet|Jumper|Bungee|"
        r"Socket|Matcha|SquidRouter|Aperture|Odos|Unizen|Dexible|"
        r"1inch\s*Fusion|Kame\s*Aggregator|Ekubo|Hashflow|ParaSwap|"
        r"Chainge\s*Finance|Rubic|Maestro|Gyro\.finance.*router|"
        r"1inch\s+aggregator\s+infrastructure)\b|"
        r"malicious calldata|arbitrary external call|arbitrary executor call",
        re.I),
    "R5_frontend_dns_hijack": re.compile(
        r"frontend\s+(?:hijack|attack|misconfig|compromise|approval)|"
        r"front[- ]end\s+(?:hijack|attack|misconfig|compromise)|"
        r"DNS\s*(?:hijack|redirect|attack|spoof)|domain\s+(?:hijack|spoof)|"
        r"malicious\s+(?:JS|JavaScript|frontend)|Cloudflare\s+(?:API|breach)|"
        r"website\s+(?:compromis|hijack|redirect)|UI\s+redirect|"
        r"social\s+media.{0,15}hijack.{0,30}sign",
        re.I),
    "R6_individual_wallet_phishing": re.compile(
        r"individual[- ]?(?:wallet|user|LP)\s+(?:phish|drain|compromise)|"
        r"single[- ]?(?:user|wallet|EOA)\s+(?:phish|drain|compromise)|"
        r"phished off[- ]?protocol|user\s+(?:was\s+)?phished|"
        r"malicious\s+Zoom|poap\.eth|personal\s+wallet|"
        r"\d+\s+bot\s+users|approval[- ]?phish|approval[- ]?signing\s+scam|"
        r"wallet\s+drain(?:ed)?|approval\s+phishing",
        re.I),
}


def depositor_facing_mask(h: pd.DataFrame) -> pd.Series:
    """True where the event is user-facing (kept), False where excluded."""
    text = (
        h["description"].fillna("")
        + " "
        + h.get("technique", pd.Series("", index=h.index)).fillna("")
        + " "
        + h["name"].fillna("")
    ).str.lower()
    excluded = pd.Series(False, index=h.index)
    for pat in _DEPOSITOR_FILTER_RULES.values():
        excluded = excluded | text.str.contains(pat)
    return ~excluded


# --- 9c. Camera-ready data pipeline ----------------------------------------

def load_camera_ready_hacks() -> pd.DataFrame:
    """Load the consolidated DeFi event set (already sector-corrected and
    non-DeFi-free, see events_consolidation.py) and apply the depositor-facing
    filter to obtain the working sample used for every headline number."""
    h = pd.read_csv(DATA / "events_consolidated.csv", parse_dates=["date"])
    h = h[(h["date"] >= ANALYSIS_WINDOW_START) &
          (h["date"] <= ANALYSIS_WINDOW_END)].copy().reset_index(drop=True)
    # `gross` is the column downstream code uses (see hacks.csv loading path).
    h["gross"] = h["loss_usd"].astype(float)
    return h.loc[depositor_facing_mask(h)].copy().reset_index(drop=True)


# --- 9d. Per-sector fitting and LDA ----------------------------------------

# Exposure-cap and TVL values matching the paper's DefiLlama June-2026 snapshot.
SECTOR_TVL_USD = {
    "Bridge": 49.1e9,
    "Lending": 66.8e9,
    "Stablecoin": 26.5e9,
    "Derivatives": 65.9e9,
    "Yield": 11.4e9,
    "DEX": 14.5e9,
    "Other": 18.2e9,
}
try:
    _caps = json.loads(
        (DATA / "raw" / "defillama" /
         "sector_exposure_caps_2026-06-30.json").read_text()
    )["caps_usd"]
    SECTOR_CAP_USD = dict(_caps)
    # Stablecoin cap is the largest single decentralized stablecoin, not the
    # aggregate USDT/USDC (which are fiat-backed and out of scope).
    try:
        _stables = json.loads(
            (DATA / "raw" / "defillama" /
             "stablecoins_2026-06-30.json").read_text()
        )

        def _coin_usd(c):
            circ = c.get("circulating") or {}
            amt = list(circ.values())[0] if circ else 0.0
            return float(amt or 0.0) * float(c.get("price") or 1.0)

        _dec = [c for c in _stables
                if (c.get("pegMechanism") or "") != "fiat-backed"]
        SECTOR_CAP_USD["Stablecoin"] = float(_coin_usd(max(_dec, key=_coin_usd)))
    except FileNotFoundError:
        pass
except FileNotFoundError:
    SECTOR_CAP_USD = {}


BASE_SEED = 42


def _sector_rng(sector: str, kind: str, base: int = BASE_SEED) -> np.random.Generator:
    """Return an independent, stable RNG stream for (sector, kind).

    SeedSequence with a deterministic per-(sector, kind) child key so
    (a) each sector's bootstrap / LDA sim is reproducible in isolation and
    (b) reordering or adding sectors doesn't perturb others' Monte-Carlo draws.
    """
    child = sum((ord(c) * (i + 1)) for i, c in enumerate(f"{sector}|{kind}"))
    return np.random.default_rng(np.random.SeedSequence([base, child]))


def run_per_sector_lda(h: pd.DataFrame) -> dict:
    """Per-sector POT-GPD + NB + compound-NB LDA on the camera-ready sample.

    Each sector uses its own seeded RNG stream (see `_sector_rng`), so
    Monte-Carlo draws are stable per sector across refactors.
    """
    results = {}
    date_min = h["date"].min()
    date_max = h["date"].max()
    years_obs = (date_max - date_min).days / 365.25
    full_months = pd.date_range(
        date_min.strftime("%Y-%m-01"), date_max, freq="MS"
    )
    for sector, tvl in SECTOR_TVL_USD.items():
        sub = h[h["sector"] == sector]
        if len(sub) < 30:
            continue
        x = sub["gross"].values.astype(float)

        plateau = select_threshold_plateau(x)
        q = plateau["q_star"]
        pot = fit_pot_gpd(x, threshold_q=q)
        pot_boot = bootstrap_pot_gpd(
            x, q, n_boot=1000, rng=_sector_rng(sector, "pot_boot"),
        )

        counts = (
            sub.set_index("date")
            .resample("MS").size()
            .reindex(full_months, fill_value=0)
            .values.astype(float)
        )
        nb = fit_negbin(counts)
        r_nb = 1.0 / nb["alpha"]
        p_nb = r_nb / (r_nb + nb["mu"])
        nb_ll = float(stats.nbinom.logpmf(counts, r_nb, p_nb).sum())
        poi_ll = float(stats.poisson.logpmf(counts, counts.mean()).sum())
        lr_vs_poi = 2.0 * (nb_ll - poi_ll)
        p_value = float(stats.chi2.sf(lr_vs_poi, df=1))

        lam = len(x) / years_obs
        cap = float(SECTOR_CAP_USD.get(sector, tvl * 0.5))
        alpha_annual = nb["alpha"] / 12.0

        # Paired capped/uncapped sim from a shared per-sector parent stream
        # so they differ only in the cap.
        parent = _sector_rng(sector, "lda")
        parent_state = parent.bit_generator.state
        samp_c = severity_sampler_pot(x, pot["threshold_usd"], pot["xi"],
                                       pot["beta"], cap=cap)
        tot_c = lda_simulate(lam, samp_c, n_years=200_000, rng=parent,
                             frequency="nb", nb_alpha=alpha_annual)
        parent_u = np.random.default_rng()
        parent_u.bit_generator.state = parent_state
        samp_u = severity_sampler_pot(x, pot["threshold_usd"], pot["xi"],
                                       pot["beta"], cap=None)
        tot_u = lda_simulate(lam, samp_u, n_years=200_000, rng=parent_u,
                             frequency="nb", nb_alpha=alpha_annual)

        mean_S = float(tot_c.mean())
        var999_c = float(np.quantile(tot_c, 0.999))
        var995_c = float(np.quantile(tot_c, 0.995))
        var99_c  = float(np.quantile(tot_c,  0.99))
        var999_u = float(np.quantile(tot_u, 0.999))

        # Parametric LDA bootstrap for IQR + CI — same distributional
        # assumption as the parametric CI on xi_hat in bootstrap_pot_gpd,
        # so Table 4 and Table 6 uncertainty are on the same footing.
        iqr = bootstrap_lda_bps_parametric(
            x, q, lam, tvl, n_boot=200, n_years=50_000,
            frequency="nb", nb_alpha=alpha_annual, cap=cap,
            rng=_sector_rng(sector, "lda_boot_param"),
        )

        results[sector] = {
            "n": int(len(x)),
            "q_star": float(q),
            "n_u": int(pot["n_exceedances"]),
            "xi": float(pot["xi"]),
            "xi_ci_lo": float(pot_boot["xi_lo"]),
            "xi_ci_hi": float(pot_boot["xi_hi"]),
            "beta_m": float(pot["beta"] / 1e6),
            "mu_monthly": float(nb["mu"]),
            "alpha_monthly": float(nb["alpha"]),
            "lr_vs_poisson": float(lr_vs_poi),
            "p_value_lr": float(p_value),
            "lambda_yr": float(lam),
            "tvl_b": float(tvl / 1e9),
            "exposure_cap_b": float(cap / 1e9),
            "mean_loss_bps": float(mean_S / tvl * 1e4),
            "var999_pct": float(var999_c / tvl * 100),
            "var995_pct": float(var995_c / tvl * 100),
            "var99_pct":  float(var99_c  / tvl * 100),
            "var999_uncapped_pct": float(var999_u / tvl * 100),
            "iqr_lo_pct": iqr["var999_bps_iqr_lo"] / 100.0,
            "iqr_hi_pct": iqr["var999_bps_iqr_hi"] / 100.0,
            "ci95_lo_pct": iqr["var999_bps_lo"] / 100.0,
            "ci95_hi_pct": iqr["var999_bps_hi"] / 100.0,
        }
    return results


# --- 9e. Report — Tables 3, 4, 5, 6 ---------------------------------------

def print_paper_report(h: pd.DataFrame, res: dict) -> None:
    sectors = ["Bridge", "Lending", "Stablecoin", "Derivatives", "Yield", "DEX", "Other"]

    print(f"AGGREGATE: n={len(h)}, gross ${h['gross'].sum()/1e6:,.1f}M, "
          f"mean ${h['gross'].mean()/1e6:.2f}M, median ${h['gross'].median()/1e6:.2f}M, "
          f"max ${h['gross'].max()/1e6:.1f}M")

    print("\n----- Table 3: sector x Basel Level-1 (USD m) -----")
    hdr = f"{'Sector':<12}{'EF':>7}{'IF':>7}{'CPBP':>7}{'EDPM':>7}{'BDSF':>7}{'Total':>8}{'n':>5}"
    print(hdr)
    tot_cat = defaultdict(int); tot_row = 0; tot_n = 0
    for sec in sectors:
        sub = h[h["sector"] == sec]
        cats = {}
        for c in ("EF", "IF", "CPBP", "EDPM", "BDSF"):
            v = int(round(sub.loc[sub["basel2_category"] == c, "gross"].sum() / 1e6))
            cats[c] = v; tot_cat[c] += v
        total = int(round(sub["gross"].sum() / 1e6))
        n = len(sub)
        tot_row += total; tot_n += n
        print(f"{sec:<12}{cats['EF']:>7}{cats['IF']:>7}{cats['CPBP']:>7}"
              f"{cats['EDPM']:>7}{cats['BDSF']:>7}{total:>8}{n:>5}")
    print(f"{'Total':<12}{tot_cat['EF']:>7}{tot_cat['IF']:>7}{tot_cat['CPBP']:>7}"
          f"{tot_cat['EDPM']:>7}{tot_cat['BDSF']:>7}{tot_row:>8}{tot_n:>5}")

    print("\n----- Table 4: POT-GPD fits -----")
    print(f"{'Sector':<12}{'n':>5}{'q*':>6}{'n_u':>5}{'xi':>7}"
          f"{'[95% CI]':>18}{'beta(M)':>9}")
    for sec in sectors:
        r = res.get(sec)
        if not r: continue
        ci = f"[{r['xi_ci_lo']:+.2f}, {r['xi_ci_hi']:+.2f}]"
        print(f"{sec:<12}{r['n']:>5}{r['q_star']:>6.2f}{r['n_u']:>5}"
              f"{r['xi']:>+7.2f}{ci:>18}{r['beta_m']:>9.1f}")

    print("\n----- Table 5: NB frequency (monthly) -----")
    print(f"{'Sector':<12}{'n':>5}{'mu/mo':>8}{'alpha':>8}{'LR':>8}{'p':>10}")
    for sec in sectors:
        r = res.get(sec)
        if not r: continue
        p = r["p_value_lr"]
        # 3 dp down to 0.01, then 4 dp, then scientific: the paper's Table 5
        # quotes this column verbatim, so the printed value has to carry
        # enough digits to be transcribed without rounding twice.
        p_fmt = (f"{p:.3f}" if p >= 1e-2 else
                 f"{p:.4f}" if p >= 1e-3 else f"{p:.1e}")
        print(f"{sec:<12}{r['n']:>5}{r['mu_monthly']:>8.2f}"
              f"{r['alpha_monthly']:>8.2f}{r['lr_vs_poisson']:>8.1f}{p_fmt:>10}")

    print("\n----- Table 6: LDA per-sector capital "
          "(IQR from parametric bootstrap) -----")
    print(f"{'Sector':<12}{'n':>5}{'lam/yr':>8}{'alpha':>7}{'xi':>7}"
          f"{'TVL(B)':>8}{'VaR99.9%':>10}{'VaR99.5%':>10}{'VaR99%':>9}"
          f"{'VaR99.9U%':>12}{'IQR%':>17}{'CI95%':>18}")
    for sec in sectors:
        r = res.get(sec)
        if not r: continue
        iqr = f"[{r['iqr_lo_pct']:.1f}, {r['iqr_hi_pct']:.1f}]"
        ci  = f"[{r['ci95_lo_pct']:.1f}, {r['ci95_hi_pct']:.1f}]"
        print(f"{sec:<12}{r['n']:>5}{r['lambda_yr']:>8.1f}{r['alpha_monthly']:>7.2f}"
              f"{r['xi']:>+7.2f}{r['tvl_b']:>8.1f}{r['var999_pct']:>9.2f}%"
              f"{r['var995_pct']:>9.2f}%{r['var99_pct']:>8.2f}%"
              f"{r['var999_uncapped_pct']:>11.2f}%{iqr:>17}{ci:>18}")


# --- 9f. Market-discipline test — Tables 7 & 8, Mann-Whitney --------------

# Venues excluded from the top-10 selection: their headline "supply APY" is
# not a market-cleared money-market deposit yield (see _EXTENDED_TEST_EXCLUDE
# for the same rule applied to the wider robustness sample).
_TOP10_EXCLUDE = ("Maple", "Sky Lending", "USDD")

# Buffer disclosures and 30-day supply APY are NOT in the DefiLlama TVL feed --
# buffers are read off each protocol's own reserve contract or governance
# disclosure, APY off its market page -- so they stay pinned to the paper's
# June-2026 snapshot. (buffer_b_or_None, supply_apy_pct)
_LENDING_BUFFER_APY = {
    "Aave V3":         (0.36, 3.36),
    "Morpho Blue":     (None, 4.60),
    "SparkLend":       (0.08, 2.84),
    "JustLend V1":     (None, 3.42),
    "Compound V3":     (0.04, 2.85),
    "Venus Core Pool": (0.01, 2.09),
    "Kamino Lend":     (None, 2.91),
    "Jupiter Lend":    (None, 4.64),
    "Fluid Lending":   (None, 6.27),
    "Euler V2":        (None, 3.59),
}


def _load_top10_lending() -> list[tuple]:
    """(name, net_tvl_b, buffer_b_or_None, supply_apy_pct) for the ten largest
    Lending venues, TVL read from the DefiLlama snapshot rather than restated.

    Selection is by GROSS supplied TVL (net + active loans), the ranking basis
    DefiLlama reports; exposure is each venue's NET TVL, the idle funds a
    contract actually holds and therefore the amount an exploit can drain.
    Reading both from the same file keeps this path and
    `top_n_lending_adequacy` on one set of numbers -- they previously
    disagreed, because the TVLs here had been transcribed by hand.
    """
    rows = json.loads(
        (DATA / "raw" / "defillama" /
         "lending_gross_tvl_2026-06-30.json").read_text()
    )
    keep = [r for r in rows if r["name"] not in _TOP10_EXCLUDE]
    top = sorted(keep, key=lambda r: -r["gross_usd"])[:10]
    missing = [r["name"] for r in top if r["name"] not in _LENDING_BUFFER_APY]
    if missing:
        raise KeyError(f"no buffer/APY snapshot for {missing}")
    # Order by exposure so the tables read largest-first on the basis the
    # model actually uses.
    top.sort(key=lambda r: -r["net_usd"])
    return [(r["name"], r["net_usd"] / 1e9, *_LENDING_BUFFER_APY[r["name"]])
            for r in top]


TOP10_LENDING = _load_top10_lending()

TBILL_PCT = 3.70

# Extended premium-test venue rules (robustness: all Lending venues >= USD 100m
# net TVL, vs the top 10). Excluded venues' headline "supply APY" is not a
# market-cleared money-market deposit yield:
_EXTENDED_TEST_EXCLUDE = {
    "Sky Lending",   # Sky/DAI Savings Rate (sUSDS, sDAI): a governance-set
                     # savings rate, not overcollateralized-borrowing supply.
    "Maple",         # private/institutional credit, not a permissionless market.
}
# Venues holding an operational-risk buffer among the >= 100m set: the four
# buffered top-10 venues plus Aave V4 (inherits the Aave DAO Umbrella).
_EXTENDED_TEST_BUFFERED = {
    "Aave V3", "SparkLend", "Compound V3", "Venus Core Pool", "Aave V4",
}


def extended_lending_premium_test(tbill_pct: float = TBILL_PCT) -> dict:
    """Robustness extension of the buffered-vs-unbuffered premium test to all
    Lending venues with net TVL >= USD 100m (vs the top 10 only), from the
    frozen DefiLlama snapshot. Savings-rate (Sky) and private-credit (Maple)
    venues are excluded; their supply APY is not a market-cleared yield. Aave V4
    is the only buffered venue among the additions."""
    lend = json.loads((DATA / "raw" / "defillama" /
                       "lending_gross_tvl_2026-06-30.json").read_text())
    slugs = {v["slug"]: v["name"] for v in lend if v["net_usd"] >= 100e6}
    apy = per_protocol_supply_apy(load_defillama_yields(), slugs)
    apy = apy[apy["apy_total_pct"].notna() & (apy["apy_total_pct"] > 0)]
    apy = apy[~apy["name"].isin(_EXTENDED_TEST_EXCLUDE)]
    buf, unbuf = [], []
    for _, r in apy.iterrows():
        spread = (r["apy_total_pct"] - tbill_pct) * 100  # bps
        (buf if r["name"] in _EXTENDED_TEST_BUFFERED else unbuf).append(spread)
    mw = stats.mannwhitneyu(buf, unbuf, alternative="less")
    return {"n": len(apy), "n_buffered": len(buf), "n_unbuffered": len(unbuf),
            "mw_u": float(mw.statistic), "mw_p": float(mw.pvalue)}


def _per_protocol_simulate(h: pd.DataFrame, sector: str,
                             sector_res: dict, tvl_p_b: float,
                             n_years: int = 1_000_000) -> tuple[float, float]:
    """Per-protocol E[S] and VaR99.9 via direct LDA simulation (paper §3):
    protocol event rate = (TVL_p / TVL_s) * lambda_s; severity is the same
    sector POT-GPD mixture; each event is capped at the protocol's own TVL.
    """
    sub = h[h["sector"] == sector]
    x = sub["gross"].values.astype(float)
    q = sector_res["q_star"]
    pot = fit_pot_gpd(x, threshold_q=q)
    share = (tvl_p_b * 1e9) / (sector_res["tvl_b"] * 1e9)
    lam_p = share * sector_res["lambda_yr"]
    alpha_annual = sector_res["alpha_monthly"] / 12.0
    cap_p = tvl_p_b * 1e9
    rng = _sector_rng(f"{sector}|{tvl_p_b}", "per_protocol_lda")
    samp = severity_sampler_pot(x, pot["threshold_usd"], pot["xi"],
                                 pot["beta"], cap=cap_p)
    tot = lda_simulate(lam_p, samp, n_years=n_years, rng=rng,
                       frequency="nb", nb_alpha=alpha_annual)
    return float(tot.mean() / 1e9), float(np.quantile(tot, 0.999) / 1e9)


def print_market_discipline(h: pd.DataFrame, res: dict) -> None:
    lend = res.get("Lending")
    if lend is None:
        print("\n[market discipline] Lending fit missing; skipping.")
        return
    # Simulate once per venue so Tables 7 & 8 share the same numbers.
    sim = {name: _per_protocol_simulate(h, "Lending", lend, tvl_b)
           for name, tvl_b, _, _ in TOP10_LENDING}

    print("\n----- Table 7: buffered Lending venues, adequacy -----")
    print(f"{'Protocol':<18}{'TVL(B)':>8}{'Buf(B)':>8}"
          f"{'E[S](B)':>10}{'buf/ES%':>9}{'VaR(B)':>10}{'buf/VaR%':>10}")
    for name, tvl_b, buf_b, _ in TOP10_LENDING:
        if buf_b is None:
            continue
        es_b, var_b = sim[name]
        print(f"{name:<18}{tvl_b:>8.2f}{buf_b:>8.2f}"
              f"{es_b:>10.2f}{buf_b/es_b*100:>8.0f}%"
              f"{var_b:>10.2f}{buf_b/var_b*100:>9.0f}%")

    print("\n----- Table 8: risk premium, top-10 Lending -----")
    print(f"{'Protocol':<18}{'buf?':>5}{'APY%':>7}{'prem(bps)':>11}"
          f"{'E[S] bps':>10}{'VaR(bps)':>11}{'prem/VaR%':>11}{'prem/ES%':>10}")
    unbuf_prem, buf_prem = [], []
    for name, tvl_b, buf_b, apy in TOP10_LENDING:
        prem = (apy - TBILL_PCT) * 100  # bps
        es_b, var_b = sim[name]
        es_bps  = es_b  / tvl_b * 1e4
        var_bps = var_b / tvl_b * 1e4
        tag = "buf" if buf_b else "unbuf"
        (buf_prem if buf_b else unbuf_prem).append(prem)
        print(f"{name:<18}{tag:>5}{apy:>7.2f}{prem:>+11.0f}"
              f"{es_bps:>10.1f}{var_bps:>11.0f}{prem/var_bps*100:>+10.1f}%"
              f"{prem/es_bps*100:>+9.0f}%")

    unbuf_prem = np.array(unbuf_prem); buf_prem = np.array(buf_prem)
    mw10 = stats.mannwhitneyu(unbuf_prem, buf_prem, alternative="greater")
    print("\nBuffered vs unbuffered (n={n1}+{n2}):"
          .format(n1=len(unbuf_prem), n2=len(buf_prem)))
    print(f"  unbuffered  mean={unbuf_prem.mean():+.1f} bps, "
          f"median={np.median(unbuf_prem):+.1f} bps")
    print(f"  buffered    mean={buf_prem.mean():+.1f} bps, "
          f"median={np.median(buf_prem):+.1f} bps")
    print(f"  median gap  = {(np.median(unbuf_prem)-np.median(buf_prem)):+.0f} bps")
    print(f"  Mann-Whitney (one-sided, unbuf > buf): "
          f"U={mw10.statistic:.0f}, p={mw10.pvalue:.3f}")

    # Robustness: extend the test to all Lending venues >= USD 100m net TVL.
    ext = extended_lending_premium_test()
    print(f"\n  Extended to all Lending venues >= USD 100m net TVL "
          f"(excl. savings-rate/credit):")
    print(f"    n={ext['n']} ({ext['n_buffered']} buffered, "
          f"{ext['n_unbuffered']} unbuffered), one-sided Mann-Whitney "
          f"U={ext['mw_u']:.0f}, p={ext['mw_p']:.3f}")


# --- 9g. §5 robustness ------------------------------------------------------

def print_robustness(h: pd.DataFrame, res: dict) -> None:
    """Reproduce the §5 Robustness paragraph:
       - Lending xi under drop-Vires and drop-Vires+BXH.
       - Per-sector tail sensitivity to drop-top-N% for N in {0.1, 0.5, 1, 2, 5}.
       - Cross-sector monthly-count dispersion (D, Cameron-Trivedi z).
       - Vuong statistic and p-value (GPD vs lognormal) at each sector's u*.
       - Buffered-venue mean cap coverage at 99.9% / 99.5% / 99.0%.
    """
    lend = h[h["sector"] == "Lending"]

    def _refit(x):
        if len(x) < 30:
            return None
        plateau = select_threshold_plateau(x)
        pot = fit_pot_gpd(x, threshold_q=plateau["q_star"])
        return {"q": plateau["q_star"], "n_u": pot["n_exceedances"],
                "xi": pot["xi"], "n": len(x)}

    print("\n----- Robustness: Lending — named-event drops -----")
    def _print_row(label, r):
        if r is None:
            print(f"  {label:<38}   (insufficient n)")
            return
        print(f"  {label:<38} n={r['n']:>4}  q*={r['q']:.2f}  "
              f"n_u={r['n_u']:>3}  xi={r['xi']:+.3f}")
    _print_row("Full Lending",
               _refit(lend["gross"].values.astype(float)))
    keep_v = ~lend["name"].str.contains("Vires", case=False, na=False)
    _print_row("Drop Vires ($515m, IF)",
               _refit(lend.loc[keep_v, "gross"].values.astype(float)))
    keep_vb = ~lend["name"].str.contains("Vires|Boy X Highspeed|BXH",
                                         case=False, na=False, regex=True)
    _print_row("Drop Vires + BXH",
               _refit(lend.loc[keep_vb, "gross"].values.astype(float)))

    print("\n----- Robustness: per-sector tail sensitivity to "
          "drop-top-N% -----")
    pcts = (0.001, 0.005, 0.01, 0.02, 0.05)
    header = f"  {'Sector':<12}{'n':>4}{'xi_full':>10}"
    for p in pcts:
        header += f"{'drop-'+f'{p*100:g}'+'%':>15}"
    print(header)
    for sec in ("Bridge", "Lending", "Stablecoin", "Derivatives",
                "Yield", "DEX", "Other"):
        x = h.loc[h["sector"] == sec, "gross"].values.astype(float)
        xi_full = _refit(x)
        cells = [xi_full]
        for pct in pcts:
            k = max(1, int(np.ceil(pct * len(x))))
            x_trim = np.sort(x)[:-k]
            r = _refit(x_trim)
            r["k"] = k
            cells.append(r)
        def _fmt(r, first=False):
            if r is None:
                return "  n/a       "
            if first:
                return f"{r['xi']:+.3f}(n_u={r['n_u']:>2})"
            return f"{r['xi']:+.3f}(k={r['k']:>2},n_u={r['n_u']:>2})"
        row = f"  {sec:<12}{len(x):>4}{_fmt(cells[0], first=True):>10}"
        for c in cells[1:]:
            row += f"{_fmt(c):>15}"
        print(row)

    print("\n----- Robustness: cross-sector count dispersion -----")
    full = pd.date_range(h["date"].min().strftime("%Y-%m-01"),
                          h["date"].max(), freq="MS")
    counts = (h.set_index("date").resample("MS").size()
              .reindex(full, fill_value=0).values.astype(float))
    d = dispersion_test(counts)
    print(f"  mean={d['mean']:.2f}, var={d['var']:.2f}, "
          f"D={d['D']:.2f}, z={d['z']:.1f}, p_overdisp={d['p_overdisp']:.1e}")

    print("\n----- Robustness: Table 4 last column at u* "
          "(Vuong GPD vs lognormal) -----")
    print(f"  {'Sector':<12}{'n_u':>5}{'ll_GPD':>10}{'ll_LN':>10}"
          f"{'V':>8}{'p':>8}  verdict")
    for sec, r in res.items():
        x = h.loc[h["sector"] == sec, "gross"].values.astype(float)
        u = float(np.quantile(x, r["q_star"]))
        v = vuong_gpd_lognormal(x, u)
        if not np.isfinite(v["V"]):
            continue
        # |V| > 1.96 separates the two models at 5%; below that the sample
        # does not distinguish them and the sign carries no information.
        verdict = (("GPD" if v["V"] > 0 else "lognormal")
                   if v["p"] < 0.05 else "tie")
        print(f"  {sec:<12}{v['n_u']:>5}{v['ll_gpd']:>10.1f}{v['ll_ln']:>10.1f}"
              f"{v['V']:>+8.2f}{v['p']:>8.3f}  {verdict}")

    print("\n----- Robustness: buffered-venue mean cap coverage at "
          "99.9% / 99.5% / 99.0% -----")
    lend_r = res.get("Lending")
    if lend_r is not None:
        sub = h[h["sector"] == "Lending"]
        x = sub["gross"].values.astype(float)
        pot = fit_pot_gpd(x, threshold_q=lend_r["q_star"])
        alpha_annual = lend_r["alpha_monthly"] / 12.0
        buffered = [(name, tvl_b, buf_b)
                    for name, tvl_b, buf_b, _ in TOP10_LENDING
                    if buf_b is not None]
        for level_pct, quant in (("99.9%", 0.999),
                                  ("99.5%", 0.995),
                                  ("99.0%", 0.99)):
            covs = []
            for name, tvl_b, buf_b in buffered:
                share = tvl_b / lend_r["tvl_b"]
                lam_p = share * lend_r["lambda_yr"]
                cap_p = tvl_b * 1e9
                rng = _sector_rng(f"Lending|{name}|{level_pct}",
                                  "cap_coverage")
                samp = severity_sampler_pot(x, pot["threshold_usd"],
                                             pot["xi"], pot["beta"],
                                             cap=cap_p)
                tot = lda_simulate(lam_p, samp, n_years=200_000, rng=rng,
                                   frequency="nb", nb_alpha=alpha_annual)
                var_b = float(np.quantile(tot, quant) / 1e9)
                covs.append(buf_b / var_b * 100)
            print(f"  VaR{level_pct}: buffered-venue mean buffer/VaR = "
                  f"{np.mean(covs):.1f}%  (per-venue: "
                  f"{', '.join(f'{c:.1f}%' for c in covs)})")


# --- 9h. Companion figures --------------------------------------------------

def _plot_xi_threshold_stability(losses: dict, fp) -> None:
    """Per-sector plateau-stability diagnostic: xi_hat as a function of the
    threshold quantile q on {0.50, 0.55, ..., 0.90}, with q* marked."""
    qs = np.round(np.arange(0.50, 0.91, 0.05), 2)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for sec, x in losses.items():
        xis = []
        for q in qs:
            u = float(np.quantile(x, q))
            e = x[x > u] - u
            if len(e) < 8:
                xis.append(np.nan); continue
            xi, _, _ = gpd_mle(e)
            xis.append(xi if np.isfinite(xi) else np.nan)
        ax.plot(qs, xis, marker="o", label=sec)
        sel = select_threshold_plateau(x)
        ax.axvline(sel["q_star"], color="grey", alpha=0.15)
    ax.set_xlabel(r"threshold quantile $q$")
    ax.set_ylabel(r"$\hat\xi(q)$")
    ax.axhline(0.0, color="k", linewidth=0.5)
    ax.axhline(1.0, color="k", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_title("POT-GPD tail-index stability across thresholds")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout(); fig.savefig(fp, dpi=140); plt.close(fig)


def main() -> None:
    """Single results entry point: filter + fit + all tables + robustness,
    then the figures and machine-readable risk summary. One invocation
    (`python code.py`) regenerates every artifact the paper cites."""
    h = load_camera_ready_hacks()
    res = run_per_sector_lda(h)
    print_paper_report(h, res)
    print_market_discipline(h, res)
    print_robustness(h, res)
    print("\n===== Figures and risk summary =====")
    regenerate_figures_and_summary()


if __name__ == "__main__":
    main()
