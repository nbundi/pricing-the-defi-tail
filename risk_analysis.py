"""Quantitative risk analysis of DeFi exploit losses.

Inputs:  output/hacks.csv          (produced by main.py)
         output/sector_tvl_panel.csv

Outputs: figures/r*_*.png          (new EVT / LDA / Hawkes / FN figures)
         output/risk_summary.json  (machine-readable headline numbers)

Methods applied:
    1. Severity tail analysis: Hill, Pickands, mean-excess, POT-GPD MLE,
       GEV block maxima, lognormal/Weibull MLE, Clauset power-law MLE+KS,
       Vuong-style log-likelihood ratio test vs lognormal alternative.
    2. Frequency model: monthly Poisson vs negative-binomial, dispersion test.
    3. Loss-distribution approach (LDA): compound Poisson-GPD Monte Carlo
       of annual aggregate losses; VaR/ES at 95/99/99.5/99.9%.
    4. Temporal clustering: interarrival KS vs exponential, Fano factor,
       monthly-count autocorrelation, simple univariate Hawkes MLE.
    5. F-N curve (annual frequency of exceedance): DeFi sectors with
       reference benchmarks from operational risk, NatCat, and process safety.
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

# Analysis window. Data are collected back to 2016, but the analysis is
# restricted to 2020-01-01 onward because the DeFi ecosystem was
# essentially non-existent before then (1 event in 2016, 1 in 2017, 0 in
# 2018-2019). Including those years biases the frequency model.
ANALYSIS_WINDOW_START = pd.Timestamp("2020-01-01")


def load_hacks(full_window: bool = False) -> pd.DataFrame:
    """Load the canonical multi-source incident master CSV produced by
    ingest_consolidated.py. By default filters to ANALYSIS_WINDOW_START
    onward; pass `full_window=True` to retain the 2011-onward records
    (used by the §4 events-over-time scatter)."""
    fp = DATA / "events_consolidated.csv"
    if not fp.exists():
        raise FileNotFoundError(
            f"Canonical event dataset not found at {fp}. "
            f"Run `python ingest_consolidated.py` first.")
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
        mask &= h["date"] >= ANALYSIS_WINDOW_START
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


def pickands_estimator(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pickands (1975) estimator for the EVT shape parameter."""
    xs = np.sort(x)[::-1]
    n = len(xs)
    ks = np.arange(4, n // 4 + 1)
    xis = []
    for k in ks:
        num = xs[k - 1] - xs[2 * k - 1]
        den = xs[2 * k - 1] - xs[4 * k - 1]
        if den <= 0 or num <= 0:
            xis.append(np.nan)
        else:
            xis.append(math.log(num / den) / math.log(2))
    return ks, np.array(xis)


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
# 2. Distribution fits (lognormal, Weibull, Pareto, power-law)
# ---------------------------------------------------------------------------

def fit_lognormal(x):
    mu = np.log(x).mean()
    sigma = np.log(x).std(ddof=0)
    ll = stats.lognorm.logpdf(x, s=sigma, scale=math.exp(mu)).sum()
    return {"mu": float(mu), "sigma": float(sigma), "loglik": float(ll)}


def fit_pareto(x):
    """Classical Pareto (single-parameter, x >= xmin) MLE."""
    xmin = float(x.min())
    alpha = len(x) / (np.log(x / xmin).sum())
    # density: alpha * xmin^alpha / x^(alpha+1)
    ll = (len(x) * math.log(alpha) + len(x) * alpha * math.log(xmin)
          - (alpha + 1) * np.log(x).sum())
    return {"alpha": float(alpha), "xmin": xmin, "loglik": float(ll)}


def fit_weibull(x):
    shape, loc, scale = stats.weibull_min.fit(x, floc=0)
    ll = stats.weibull_min.logpdf(x, shape, loc=loc, scale=scale).sum()
    return {"shape": float(shape), "scale": float(scale), "loglik": float(ll)}


def clauset_power_law(x: np.ndarray) -> dict:
    """Clauset-Shalizi-Newman (2009) power-law MLE with x_min via KS minimization.

    For x >= xmin, model is f(x) = (alpha-1) / xmin * (x/xmin)^(-alpha).
    Continuous MLE: alpha_hat = 1 + n / sum(log(x/xmin)).

    Then pick xmin that minimizes the KS distance between empirical CDF on
    x >= xmin and the fitted power-law CDF.
    """
    xs_all = np.sort(x)
    # candidate xmins = data points (skip very small ones for stability)
    candidates = np.unique(xs_all)
    candidates = candidates[(candidates >= np.quantile(xs_all, 0.05)) &
                            (candidates <= np.quantile(xs_all, 0.97))]
    best = {"ks": np.inf}
    for xmin in candidates:
        tail = xs_all[xs_all >= xmin]
        n = len(tail)
        if n < 20:
            continue
        lr = np.log(tail / xmin)
        s = lr.sum()
        if s <= 0:
            continue
        alpha = 1 + n / s
        if alpha <= 1:
            continue
        # empirical CDF
        ecdf = np.arange(1, n + 1) / n
        # theoretical CDF for power-law tail
        tcdf = 1 - (tail / xmin) ** (1 - alpha)
        ks = np.max(np.abs(ecdf - tcdf))
        if ks < best["ks"]:
            best = {"alpha": float(alpha), "xmin": float(xmin), "n_tail": n,
                    "ks": float(ks)}
    return best


def loglik_powerlaw(x, alpha, xmin):
    tail = x[x >= xmin]
    return (len(tail) * (math.log(alpha - 1) - math.log(xmin))
            - alpha * np.log(tail / xmin).sum())


def loglik_lognormal_tail(x, xmin, mu, sigma):
    """Lognormal restricted to x >= xmin: pdf / (1 - F(xmin))."""
    tail = x[x >= xmin]
    p_above = 1 - stats.lognorm.cdf(xmin, s=sigma, scale=math.exp(mu))
    if p_above <= 0:
        return -np.inf
    return stats.lognorm.logpdf(tail, s=sigma, scale=math.exp(mu)).sum() - len(tail) * math.log(p_above)


def vuong_test(ll1, ll2, n):
    """One-sided Vuong-style likelihood-ratio test statistic on a per-obs basis.

    Returns (LR, R, p) where LR = ll1 - ll2 (positive favors model 1),
    R = normalized statistic, p is two-sided p-value.
    """
    lr = ll1 - ll2
    # Crude approximation; not full Vuong with variance estimate.
    return lr


# ---------------------------------------------------------------------------
# 3. Frequency model
# ---------------------------------------------------------------------------

def monthly_counts(h: pd.DataFrame, start: str | None = None) -> pd.Series:
    """Monthly hack counts over [start, last]. If start is None, uses the
    full window from the first hack to the last (period 2016-2026 in our
    dataset)."""
    if start is None:
        start = h["date"].min().strftime("%Y-%m-01")
    h2 = h[h["date"] >= start]
    counts = h2.set_index("date").resample("MS").size()
    full = pd.date_range(start, h["date"].max(), freq="MS")
    return counts.reindex(full, fill_value=0)


def annual_counts(h: pd.DataFrame) -> pd.Series:
    """Annual hack counts; non-stationarity diagnostic."""
    return h.set_index("date").resample("YS").size()


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
                 rng=None) -> np.ndarray:
    """Compound Poisson Monte Carlo of annual aggregate losses."""
    if rng is None:
        rng = np.random.default_rng()
    Ns = rng.poisson(annual_lambda, size=n_years)
    totals = np.zeros(n_years)
    # vectorized severity per year
    flat_N = Ns.sum()
    sev = severity_sampler(flat_N, rng)
    idx = 0
    for i, n in enumerate(Ns):
        if n == 0:
            continue
        totals[i] = sev[idx:idx + n].sum()
        idx += n
    return totals


def severity_sampler_pot(body_x: np.ndarray, threshold: float,
                         xi: float, beta: float):
    """Mixture sampler: with prob p_below sample from empirical CDF below
    threshold (bootstrap), else sample from GPD(xi, beta) + threshold."""
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
        return out
    return _sample


# ---------------------------------------------------------------------------
# 5. Temporal / clustering analysis
# ---------------------------------------------------------------------------

def fit_hawkes_exp(times: np.ndarray, T: float) -> dict:
    """Univariate exponential Hawkes MLE on a single window [0, T].

    Conditional intensity:  lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta (t - t_i))

    Log-likelihood (Ozaki 1979):
        L = sum_i log lambda(t_i) - integral_0^T lambda(s) ds
          = sum_i log(mu + alpha * A(i))  -  mu * T
            - (alpha/beta) * sum_i (1 - exp(-beta (T - t_i)))
    where A(i) = sum_{j<i} exp(-beta (t_i - t_j)) computed recursively.

    Stationarity requires alpha < beta (branching ratio n = alpha/beta < 1).
    """
    n = len(times)
    if n < 30:
        return {"converged": False}
    dt = np.diff(times)

    # Constrain `beta` to half-life >= 1 day (beta <= ln 2 / 1). The input
    # timestamps are date-resolution with sub-day jitter, so any signal at
    # sub-day timescales is an artefact of how multi-event days were
    # resolved into a point process. We want cross-day clustering only.
    BETA_MAX = math.log(2)  # half-life of 1 day

    def neg_ll(params):
        log_mu, log_alpha, log_beta = params
        mu = math.exp(log_mu)
        alpha = math.exp(log_alpha)
        beta = math.exp(log_beta)
        if beta > BETA_MAX:                # constraint via penalty
            return 1e10
        A = np.empty(n)
        A[0] = 0.0
        for i in range(1, n):
            A[i] = math.exp(-beta * dt[i - 1]) * (1 + A[i - 1])
        lam = mu + alpha * A
        if (lam <= 0).any():
            return 1e10
        compensator = mu * T + (alpha / beta) * np.sum(
            1 - np.exp(-beta * (T - times)))
        ll = np.log(lam).sum() - compensator
        return -ll

    # multi-start: cover (a) fast-decay/low-branching regimes,
    # (b) slow-decay/high-branching regimes, and (c) the seismic ETAS
    # half-life range (days–months) that's been our best estimate so far.
    base_rate = n / T
    log_base = math.log(max(base_rate, 1e-4))
    starts = [
        (log_base + math.log(0.5),  math.log(0.5),  math.log(1.0)),
        (log_base + math.log(0.7),  math.log(0.1),  math.log(0.5)),
        (log_base + math.log(0.3),  math.log(1.0),  math.log(2.0)),
        # Seismic-ETAS-style starting points (slow decay, mid branching)
        (log_base + math.log(0.5),  math.log(0.01), math.log(0.02)),
        (log_base + math.log(0.5),  math.log(0.005),math.log(0.01)),
        # Very-slow-decay regime (relevant after the 2020 regime shift)
        (log_base + math.log(0.4),  math.log(0.02), math.log(0.04)),
    ]
    best = None
    for s in starts:
        try:
            res = optimize.minimize(neg_ll, s, method="Nelder-Mead",
                                    options={"xatol": 1e-7, "fatol": 1e-7,
                                             "maxiter": 12000})
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            pass
    if best is None or not np.isfinite(best.fun):
        return {"converged": False}
    mu = math.exp(best.x[0]); alpha = math.exp(best.x[1]); beta = math.exp(best.x[2])
    branching = alpha / beta
    half_life_days = math.log(2) / beta if beta > 0 else np.inf
    return {"converged": True, "mu": mu, "alpha": alpha, "beta": beta,
            "branching_ratio": branching, "half_life_days": half_life_days,
            "loglik": float(-best.fun)}


def poisson_loglik(times: np.ndarray, T: float) -> float:
    """Homogeneous Poisson loglik with rate = n/T (MLE)."""
    n = len(times)
    rate = n / T
    return n * math.log(rate) - rate * T


# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------

PALETTE = {
    # Pooled
    "DeFi":                       "black",
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


def plot_interarrival(times_days: np.ndarray, fp: Path):
    """Interarrival times: empirical CDF vs exponential null."""
    iat = np.diff(np.sort(times_days))
    iat = iat[iat > 0]
    lam = 1 / iat.mean()
    xs = np.sort(iat)
    ecdf = np.arange(1, len(xs) + 1) / len(xs)
    fig, (ax_cdf, ax_qq) = plt.subplots(1, 2, figsize=(13, 5.5))
    ax_cdf.step(xs, ecdf, where="post", lw=1.8, color="#3a78c2",
                label="Empirical CDF")
    ax_cdf.plot(xs, 1 - np.exp(-lam * xs), color="#c2553a", lw=1.8,
                label=f"Exponential CDF (1/λ={1/lam:.2f} days)")
    ax_cdf.set_xscale("log")
    ax_cdf.set_xlabel("Interarrival time (days, log)")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_title("Hack interarrival times vs Poisson (exponential) null")
    ax_cdf.legend(fontsize=9)
    ax_cdf.grid(True, alpha=0.3, which="both")

    # Q-Q
    theo = -np.log(1 - (np.arange(1, len(xs) + 1) - 0.5) / len(xs)) / lam
    ax_qq.scatter(theo, xs, alpha=0.6, s=18, color="#3a78c2")
    lim = max(theo.max(), xs.max())
    ax_qq.plot([0, lim], [0, lim], "r--", lw=1.5)
    ax_qq.set_xlabel("Exponential theoretical quantile (days)")
    ax_qq.set_ylabel("Empirical interarrival (days)")
    ax_qq.set_title("Q-Q vs exponential — deviations indicate non-Poisson timing")
    ax_qq.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fp, dpi=140); plt.close(fig)


def plot_loglog_ccdf(losses: dict[str, np.ndarray], pl_fits: dict[str, dict],
                     fp: Path, axis: str = "sector"):
    """Log-log complementary CDF with power-law fit overlay."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, x in losses.items():
        if len(x) < 30:
            continue
        xs = np.sort(x)
        n = len(xs)
        ccdf = 1 - np.arange(1, n + 1) / (n + 1)
        ax.plot(xs, ccdf, marker="o", ms=3, lw=0,
                color=PALETTE.get(label, "gray"),
                label=f"{label} (n={n})", alpha=0.85)
        fit = pl_fits.get(label)
        if fit and fit.get("alpha"):
            xmin, alpha = fit["xmin"], fit["alpha"]
            # tail fraction
            p_above = (xs >= xmin).mean()
            xx = np.geomspace(xmin, xs.max(), 100)
            yy = p_above * (xx / xmin) ** (1 - alpha)
            ax.plot(xx, yy, color=PALETTE.get(label, "gray"), ls="--", lw=1.2,
                    alpha=0.95)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Loss L  (USD, log)")
    ax.set_ylabel("Empirical CCDF  P(X ≥ L)")
    ax.set_title(f"Log-log CCDF of DeFi exploit losses by {axis} "
                 "with Clauset MLE power-law fit\n"
                 "Dashed lines = fitted power-law tail above estimated x_min")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(fp, dpi=140); plt.close(fig)


def plot_annual_counts(counts: pd.Series, fp: Path):
    """Annual hack counts 2016-2026 — non-stationarity diagnostic."""
    fig, ax = plt.subplots(figsize=(10, 5))
    years = [d.year for d in counts.index]
    ax.bar(years, counts.values, color="#3a78c2", edgecolor="white",
           alpha=0.9, label="Annual hack count")
    mu = counts.values.mean()
    ax.axhline(mu, color="#c2553a", lw=1.5,
               label=f"Mean across all years = {mu:.1f}")
    for y, v in zip(years, counts.values):
        ax.text(y, v + 1, str(int(v)), ha="center", fontsize=9)
    ax.set_xlabel("Year")
    ax.set_ylabel("# DeFi-Protocol hacks")
    ax.set_title("Annual DeFi-Protocol exploit frequency 2016–2026\n"
                 "Strong non-stationarity: <2 per year before 2020, "
                 "55–80 per year since 2021")
    ax.set_xticks(years)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fp, dpi=140); plt.close(fig)


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
                 "— gross loss vs date, by Basel III L1 category")
    ax.legend(fontsize=8, loc="lower right", ncol=3, framealpha=0.95)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fp, dpi=140); plt.close(fig)


def plot_loss_distribution_by_sector(h: pd.DataFrame, fp: Path):
    """Violin/strip plot of log10(gross loss) by sector, ordered by
    median loss, with cross-sector medians and 90th-percentile lines
    visible."""
    sectors_order = ["Bridge", "Derivatives", "Lending", "Stablecoin",
                     "Yield", "DEX", "Other"]
    sectors_present = [s for s in sectors_order
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
    for i, body in enumerate(parts["bodies"]):
        col = PALETTE.get(sectors_present[i].replace(" ", ""), "#888888")
        col = PALETTE.get(sectors_present[i], col)
        body.set_facecolor(col)
        body.set_edgecolor("black")
        body.set_alpha(0.55)
    parts["cmedians"].set_color("black")
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


def plot_tail_index_benchmarks(defi_xi: dict[str, float], fp: Path,
                                axis: str = "sector"):
    """Bar chart of fitted tail indices for a DeFi breakdown axis
    (sector / Basel L1 / Chang) with reference bands for other
    loss-generating processes."""
    items = sorted(defi_xi.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.barh(labels, vals, color=[PALETTE.get(l, "gray") for l in labels],
                   edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(v + 0.02, b.get_y() + b.get_height() / 2,
                f"ξ̂={v:.2f}", va="center", fontsize=9)

    # Reference bands
    refs = [
        ("Equity daily returns (S&P)",     0.25, 0.40, "#a4d4ff"),
        ("Hurricane/NatCat losses",         0.50, 1.00, "#ffd4b0"),
        ("Op-Risk losses (Moscadelli 2004)",0.85, 1.20, "#ff9999"),
        ("Cyber-breach losses (Eling 2019)",0.70, 1.10, "#ffcc99"),
        ("Industrial accident fatalities",  0.50, 1.00, "#d4b0ff"),
        ("Major earthquake losses",         0.60, 1.10, "#b0d4b0"),
    ]
    for i, (lab, lo, hi, color) in enumerate(refs):
        y = len(labels) + i + 1
        ax.axhspan(y - 0.4, y + 0.4, xmin=0, xmax=1,
                   color="white")
        ax.barh(y, hi - lo, left=lo, color=color, edgecolor="gray",
                alpha=0.85, height=0.7)
        ax.text(hi + 0.02, y, f" {lab}: ξ ≈ [{lo:.1f}, {hi:.1f}]",
                va="center", fontsize=9)

    ax.axvline(0.5, color="orange", ls="--", lw=1, alpha=0.7)
    ax.axvline(1.0, color="red", ls="--", lw=1, alpha=0.7)
    ax.text(0.5, -1.2, "ξ=0.5\n(infinite variance)", color="orange",
            ha="center", fontsize=8)
    ax.text(1.0, -1.2, "ξ=1.0\n(infinite mean)", color="red",
            ha="center", fontsize=8)

    ax.set_xlim(0, 2.5)
    ax.set_xlabel("EVT shape parameter ξ  (Pareto tail index)")
    ax.set_title(f"Tail index of DeFi exploit losses by {axis} vs reference loss-generating processes\n"
                 "POT-GPD MLE; reference bands from published literature")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(fp, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Pipeline
# ---------------------------------------------------------------------------

def main():
    print("[1/7] Loading data ...")
    h = load_hacks()
    panel = load_tvl_panel()
    print(f"  in-scope DeFi-Protocol hacks: n = {len(h)}")
    print(f"  date range: {h['date'].min().date()} .. {h['date'].max().date()}")

    # Three breakdown axes — sector, Basel II Level-1, Chang (SOA)
    sectors = ["Lending", "DEX", "Bridge", "Yield", "Stablecoin", "Derivatives", "Other"]
    basel_cats = ["EF", "IF", "CPBP", "EDPM", "BDSF"]   # EPWS & DPA empty in DeFi

    def _losses_by(col: str, labels: list[str]) -> dict[str, np.ndarray]:
        # No pooled-DeFi entry: per-sector calibration only (the paper
        # does not produce DeFi-wide capital numbers).
        d = {}
        for lab in labels:
            d[lab] = h.loc[h[col] == lab, "gross"].values
        return d

    losses        = _losses_by("sector",          sectors)
    losses_basel  = _losses_by("basel2_category", basel_cats)

    # ---- 2. Severity tail fits ---------------------------------------------
    print("[2/7] Severity tail fits ...")

    def fit_axis(d: dict[str, np.ndarray]) -> dict[str, dict]:
        out = {}
        for label, x in d.items():
            if len(x) < 20:
                continue
            ln  = fit_lognormal(x)
            wb  = fit_weibull(x)
            n_loc = len(x)
            if n_loc >= 200:    q0 = 0.85
            elif n_loc >= 80:   q0 = 0.80
            elif n_loc >= 40:   q0 = 0.70
            else:               q0 = 0.60
            pot = fit_pot_gpd(x, threshold_q=q0)
            scan = {}
            for q in (0.50, 0.60, 0.70, 0.80, 0.90):
                f = fit_pot_gpd(x, threshold_q=q)
                scan[f"{q:.2f}"] = {"xi": f["xi"], "beta": f["beta"],
                                    "n_exc": f["n_exceedances"],
                                    "threshold_usd": f["threshold_usd"]}
            cl  = clauset_power_law(x)
            if cl.get("xmin"):
                xmin = cl["xmin"]
                cl["loglik_pl"] = loglik_powerlaw(x, cl["alpha"], xmin)
                cl["loglik_lognormal_tail"] = loglik_lognormal_tail(
                    x, xmin, ln["mu"], ln["sigma"])
                cl["lr_pl_vs_lognormal"] = (
                    cl["loglik_pl"] - cl["loglik_lognormal_tail"])
            out[label] = {"lognormal": ln, "weibull": wb, "pot_gpd": pot,
                          "pot_threshold_scan": scan,
                          "clauset_pl": cl, "n": int(len(x)),
                          "sum_usd": float(x.sum()),
                          "max_usd": float(x.max()),
                          "median_usd": float(np.median(x))}
            if pot.get("xi") is not None and np.isfinite(pot["xi"]):
                print(f"  {label:<26s} n={len(x):>4d}  POT-GPD ξ̂={pot['xi']:+.3f}  "
                      f"β̂=${pot['beta']/1e6:.1f}m  thresh=${pot['threshold_usd']/1e6:.1f}m  "
                      f"PL α̂={cl.get('alpha', float('nan')):.2f}  "
                      f"x_min=${cl.get('xmin', 0)/1e6:.1f}m")
        return out

    print(" -- by sector --")
    fits        = fit_axis(losses)
    print(" -- by Basel II Level-1 (computed for output JSON only) --")
    fits_basel  = fit_axis(losses_basel)

    # ---- 3. Plots: mean excess + Hill + log-log CCDF + GPD Q-Q -------------
    # The paper renders per-sector EVT diagnostics only; per-Basel results
    # are retained in the output JSON but no longer plotted (the paper
    # focuses on sector-level capital requirements).
    print("[3/7] Plotting EVT diagnostics (sector axis only) ...")

    keep_sector = {k: v for k, v in losses.items() if k in fits}
    plot_mean_excess(keep_sector, FIG / "r1_mean_excess.png", axis="sector")
    plot_hill(keep_sector, FIG / "r2_hill_plot.png", axis="sector")
    pl_fits_sector = {k: v["clauset_pl"] for k, v in fits.items()}
    plot_loglog_ccdf(keep_sector, pl_fits_sector,
                     FIG / "r3_loglog_ccdf.png", axis="sector")
    xi_sector = {k: v["pot_gpd"]["xi"] for k, v in fits.items()
                 if np.isfinite(v["pot_gpd"]["xi"])}
    plot_tail_index_benchmarks(xi_sector,
                               FIG / "r5_tail_index_benchmarks.png",
                               axis="sector")

    # Per-sector POT-GPD QQ plots — one for Lending (the headline
    # capital sector). The pooled-DeFi fit is not produced; per the
    # paper's per-sector orientation we use Lending as the diagnostic
    # exemplar.
    pot_lending = fits["Lending"]["pot_gpd"]
    plot_gpd_qq(losses["Lending"], pot_lending["xi"], pot_lending["beta"],
                pot_lending["threshold_usd"], FIG / "r4_gpd_qq_lending.png",
                title=(f"POT-GPD Q-Q  ·  DeFi-Lending  ·  n_exceed="
                       f"{pot_lending['n_exceedances']}\n"
                       f"ξ̂={pot_lending['xi']:.3f}  "
                       f"β̂=USD {pot_lending['beta']/1e6:.1f}m  "
                       f"threshold (q={pot_lending['threshold_q']:.0%}) = "
                       f"USD {pot_lending['threshold_usd']/1e6:.1f}m"))

    # ---- 4. Frequency model + monthly counts -------------------------------
    print("[4/7] Frequency model (full 2016-2026 window) ...")
    counts = monthly_counts(h, start=None)  # full window
    disp = dispersion_test(counts.values)
    nb = fit_negbin(counts.values)
    pois_ll = (stats.poisson.logpmf(counts.values, counts.values.mean())).sum()
    nb_r = 1 / nb["alpha"]; nb_p = nb_r / (nb_r + nb["mu"])
    nb_ll = stats.nbinom.logpmf(counts.values, nb_r, nb_p).sum()
    lr_stat = 2 * (nb_ll - pois_ll)
    lr_p = 1 - stats.chi2.cdf(lr_stat, df=1)
    yearly = annual_counts(h)
    print(f"  monthly mean = {disp['mean']:.2f}  var = {disp['var']:.2f}  "
          f"D = var/mean = {disp['D']:.2f}  z-overdisp = {disp['z']:.2f}  "
          f"p = {disp['p_overdisp']:.3g}")
    print(f"  NB(μ={nb['mu']:.2f}, α={nb['alpha']:.3f})  "
          f"vs Poisson LR = {lr_stat:.2f}  p = {lr_p:.3g}")
    print(f"  annual counts: {dict((str(d.year), int(v)) for d, v in yearly.items())}")
    plot_monthly_counts(counts, nb, FIG / "r6_monthly_counts.png")
    plot_annual_counts(yearly, FIG / "r6b_annual_counts.png")

    # Exploratory plots for §4 — DeFi-protocol events 2020-2026.
    # The 2020 cutoff is justified in §3.5 (only 8 sparse proto-DeFi
    # pre-2020 events; the rest are CEX / wallet / custodian records)
    # and applied uniformly across every plot and model fit in the
    # paper. We pass the post-2020-filtered DataFrame `h` (already
    # filtered by load_hacks) to all §4 plot helpers.
    plot_events_scatter(h, FIG / "r0_events_scatter.png")
    plot_loss_distribution_by_sector(h,
                                     FIG / "r0b_loss_distribution_by_sector.png")
    sector_order = ["Bridge", "Lending", "DEX", "Yield",
                    "Stablecoin", "Derivatives", "Other"]
    plot_rolling_intensity(h, FIG / "r0c_rolling_intensity_sector.png",
                           group_col="sector", order=sector_order,
                           window_days=90,
                           title_suffix=" by sector (90-day window)",
                           include_total=False)
    plot_rolling_intensity(h, FIG / "r0d_rolling_intensity_basel.png",
                           group_col="basel2_category",
                           order=["EF", "IF", "CPBP", "EDPM", "BDSF"],
                           window_days=90,
                           title_suffix=" by event type (90-day window)",
                           include_total=False)

    # Pooled DeFi LDA intentionally omitted — the paper produces
    # capital numbers per sector only. We retain the per-sector LDA
    # in 5b below.
    years_obs_full = (h["date"].max() - h["date"].min()).days / 365.25
    tvl_recent = float(panel["DeFi"].iloc[-365:].mean())

    # ---- 5. Per-sector LDA: bank-style capital requirements -------------
    # For each sector with a sensible event count and a TVL denominator
    # in the panel, fit a compound-Poisson body+GPD-tail LDA and report
    # mean / VaR99 / ES99 / ES99.9 as bps of the trailing-365d sector-
    # average TVL.
    print("[5/7] Per-sector capital-requirements LDA ...")
    sector_tvl_map = {
        "Lending":     "Lending",
        "DEX":         "DEX",
        "Bridge":      "Bridge",
        "Yield":       "Yield",
        "Derivatives": "Derivatives",
    }
    sector_lda = {}
    for sect, tvl_col in sector_tvl_map.items():
        x_s = losses.get(sect)
        if x_s is None or len(x_s) < 30:
            continue
        pot_s = fits[sect]["pot_gpd"]
        years_obs = (h["date"].max() - h["date"].min()).days / 365.25
        n_s = (h["sector"] == sect).sum()
        lam_s = n_s / years_obs
        tvl_s = float(panel[tvl_col].iloc[-365:].mean())
        samp_s = severity_sampler_pot(x_s, pot_s["threshold_usd"],
                                       pot_s["xi"], pot_s["beta"])
        tot_s = lda_simulate(lam_s, samp_s, n_years=200_000, rng=RNG)
        qs = {q: float(np.quantile(tot_s, q))
              for q in (0.5, 0.95, 0.99, 0.999)}
        mean_s = float(tot_s.mean())
        es99_s = float(tot_s[tot_s >= qs[0.99]].mean())
        es999_s = float(tot_s[tot_s >= qs[0.999]].mean())
        sector_lda[sect] = {
            "n_events": int(n_s),
            "lambda_yr": float(lam_s),
            "tvl_recent_usd": tvl_s,
            "xi": float(pot_s["xi"]),
            "beta_usd": float(pot_s["beta"]),
            "threshold_usd": float(pot_s["threshold_usd"]),
            "mean_loss_usd":     mean_s,
            "var99_usd":         qs[0.99],
            "es99_usd":          es99_s,
            "es999_usd":         es999_s,
            "mean_loss_bps":     mean_s   / tvl_s * 1e4,
            "var99_bps":         qs[0.99] / tvl_s * 1e4,
            "es99_bps":          es99_s   / tvl_s * 1e4,
            "es999_bps":         es999_s  / tvl_s * 1e4,
        }
        print(f"  {sect:<13s} n={n_s:>3d}  λ={lam_s:5.1f}/yr  "
              f"TVL=USD {tvl_s/1e9:5.1f}B  "
              f"E[L]={mean_s/1e6:6.0f}m ({mean_s/tvl_s*1e4:5.0f}bps)  "
              f"VaR99={qs[0.99]/1e6:7.0f}m ({qs[0.99]/tvl_s*1e4:5.0f}bps)  "
              f"ES99={es99_s/1e6:7.0f}m ({es99_s/tvl_s*1e4:5.0f}bps)")

    # ---- 6. Temporal clustering ------------------------------------------
    print("[6/7] Temporal clustering ...")
    t0 = h["date"].min()
    times_days = (h["date"] - t0).dt.total_seconds().values / 86400
    T = float(times_days.max())
    # Add tiny jitter for ties (multiple hacks same day)
    times_days = times_days + RNG.uniform(0, 0.5, size=len(times_days))
    times_days.sort()
    iat = np.diff(times_days)
    iat_pos = iat[iat > 0]
    ks_stat, ks_p = stats.kstest(iat_pos, "expon", args=(0, iat_pos.mean()))
    print(f"  KS interarrival vs exponential: D={ks_stat:.3f}  p={ks_p:.3g}")
    # Fano factor on monthly counts
    fano = counts.var(ddof=1) / counts.mean()
    print(f"  Fano factor (monthly) = {fano:.3f}  (Poisson => 1)")
    # Lag-1 autocorrelation
    acf1 = float(np.corrcoef(counts.values[:-1], counts.values[1:])[0, 1])
    print(f"  Lag-1 ACF of monthly counts = {acf1:.3f}")

    # Hawkes MLE on the FULL 2016-2026 window. The constant-background
    # μ averages over both the quiet early years and the more active
    # recent period; this is conservative for the branching ratio because
    # heavy clustering in 2021-2026 is what drives the self-excitation
    # signal. A 2021+ sensitivity is reported in the paper.
    times_full = times_days.copy()
    T_full = float(times_full.max() + 1.0)  # 1d past last event to be safe
    hk = fit_hawkes_exp(times_full, T_full)
    pll = poisson_loglik(times_full, T_full)
    if hk["converged"]:
        lr_hk = 2 * (hk["loglik"] - pll)
        p_hk = 1 - stats.chi2.cdf(lr_hk, df=2)
        hk["lr_vs_poisson"] = float(lr_hk)
        hk["p_vs_poisson"]  = float(p_hk)
        hk["fit_window"] = "2016-2026 full"
        print(f"  Hawkes (FULL 2016-2026): μ={hk['mu']:.4f}/day  "
              f"α={hk['alpha']:.4f}  β={hk['beta']:.4f}  "
              f"branching n={hk['branching_ratio']:.3f}  "
              f"half-life={hk['half_life_days']:.1f}d  "
              f"LR vs Poisson={lr_hk:.1f}  p={p_hk:.3g}")
    else:
        print("  Hawkes MLE (full) did not converge.")

    # Sensitivity: Hawkes fit on 2021+ only.
    h_recent = h[h["date"] >= "2021-01-01"].copy()
    t_recent = ((h_recent["date"] - pd.Timestamp("2021-01-01"))
                .dt.total_seconds().values / 86400)
    t_recent = t_recent + RNG.uniform(0, 0.5, size=len(t_recent))
    t_recent.sort()
    T_recent = float(t_recent.max() + 1)
    hk_recent = fit_hawkes_exp(t_recent, T_recent)
    if hk_recent["converged"]:
        pll_r = poisson_loglik(t_recent, T_recent)
        hk_recent["lr_vs_poisson"] = float(2 * (hk_recent["loglik"] - pll_r))
        hk_recent["fit_window"] = "2021-2026 only"
        print(f"  Hawkes (2021-2026 sensitivity): "
              f"branching n={hk_recent['branching_ratio']:.3f}  "
              f"half-life={hk_recent['half_life_days']:.1f}d")

    plot_interarrival(times_days, FIG / "r8_interarrival.png")
    years_obs = (h["date"].max() - h["date"].min()).days / 365.25

    # ---- 7. Persist summary ------------------------------------------------
    print("[7/7] Saving risk summary ...")
    summary = {
        "as_of": str(h["date"].max().date()),
        "n_hacks_in_scope": int(len(h)),
        "years_observed": float(years_obs),
        "data_source": "data/events_consolidated.csv "
                       "(DefiLlama + rekt.news + kismp123 + DeFiHackLabs + "
                       "BlockSec + de.fi/rekt-database + SlowMist Hacked; "
                       "see ingest_consolidated.py for source-precedence and "
                       "deduplication methodology)",
        "fits":       fits,
        "fits_basel": fits_basel,
        "frequency": {
            "fit_window": f"{h['date'].min().date()} .. {h['date'].max().date()}",
            "monthly_counts": {
                "mean": disp["mean"], "var": disp["var"], "D": disp["D"],
                "p_overdispersion": disp["p_overdisp"],
                "n_months": int(len(counts))},
            "annual_counts": {str(d.year): int(v) for d, v in yearly.items()},
            "neg_binomial": nb,
            "lr_nb_vs_poisson": {"stat": float(lr_stat), "p": float(lr_p)},
        },
        "lda": {
            "tvl_recent_usd_defi_total": tvl_recent,
            "per_sector": sector_lda,
        },
        "temporal": {
            "ks_interarrival_vs_exponential": {"D": float(ks_stat), "p": float(ks_p)},
            "fano_monthly": float(fano),
            "acf1_monthly": acf1,
            "hawkes_full":   hk,
            "hawkes_2021plus": hk_recent,
        },
    }
    (OUT / "risk_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote risk summary to: {OUT / 'risk_summary.json'}")
    print(f"Wrote new figures to:  {FIG}")


if __name__ == "__main__":
    main()
