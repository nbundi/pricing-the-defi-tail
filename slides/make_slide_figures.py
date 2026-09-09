#!/usr/bin/env python3
"""Slide-sized versions of the paper's two severity figures.

The figures in ../figures are drawn for the LNCS column: 2.35 in wide, 6.5 pt
ticks, sector names rotated 45 degrees to fit. Projected, that is unreadable.
This script redraws the same two panels from the same working sample -- the
depositor-facing filtered DeFi set -- in a wide slide format with upright
labels, and writes them to figures/ next to this file. It touches nothing in
../figures, so the paper's artwork is unaffected.

    python3 make_slide_figures.py
"""
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import code as paper  # noqa: E402  (the paper's pipeline, ../code.py)

OUT = Path(__file__).resolve().parent / "figures"

# Slide geometry: 2.8:1 so the panel fills the frame width and still leaves
# room for a caption line under it.
FIGSIZE = (7.7, 2.75)
AXRECT = dict(left=0.085, right=0.985, bottom=0.17, top=0.97)
FS_TICK, FS_LABEL, FS_LEGEND = 9.5, 10.5, 8.5


def working_sample():
    """The same sample the paper's figures are drawn on."""
    h = paper.load_hacks()
    h = h.loc[paper.depositor_facing_mask(h)].reset_index(drop=True)
    sectors = sorted(
        [s for s in ["Lending", "DEX", "Bridge", "Yield",
                     "Stablecoin", "Derivatives", "Other"]
         if (h["sector"] == s).any()],
        key=lambda s: -h.loc[h["sector"] == s, "gross"].median(),
    )
    losses = {s: h.loc[h["sector"] == s, "gross"].values for s in sectors}
    return h, sectors, losses


def gpd_fits(losses):
    """Plateau-selected POT-GPD per sector, as in the paper."""
    out = {}
    for s, x in losses.items():
        if len(x) < 20:
            continue
        sel = paper.select_threshold_plateau(x, tau=0.30, n_u_abs=20,
                                             n_u_frac=0.10)
        out[s] = paper.fit_pot_gpd(x, threshold_q=sel["q_star"])
    return out


def violin(h, sectors, fp):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(**AXRECT)
    groups = [np.log10(h.loc[h["sector"] == s, "gross"].values) for s in sectors]
    parts = ax.violinplot(groups, showmeans=False, showmedians=True, widths=0.8)
    for body in parts["bodies"]:
        body.set_facecolor("0.8"); body.set_edgecolor("black")
        body.set_linewidth(0.6); body.set_alpha(0.7)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("black"); parts[key].set_linewidth(0.6)
    parts["cmedians"].set_linewidth(1.0)
    rng = np.random.default_rng(0)
    for i, x in enumerate(groups, 1):
        ax.scatter(i + rng.uniform(-0.15, 0.15, size=len(x)), x,
                   s=2.0, color="black", alpha=0.25)
    ax.set_xticks(range(1, len(sectors) + 1))
    # Upright, not rotated: at slide width there is room for the sector names.
    ax.set_xticklabels(sectors, fontsize=FS_TICK)
    decades = [(3, "1k"), (4, "10k"), (5, "100k"), (6, "1m"),
               (7, "10m"), (8, "100m"), (9, "1B")]
    ax.set_yticks([lvl for lvl, _ in decades])
    ax.set_yticklabels([lab for _, lab in decades], fontsize=FS_TICK)
    ax.set_ylabel("Gross loss  (USD, log)", fontsize=FS_LABEL)
    ax.set_xlim(0.4, len(sectors) + 0.6)
    ax.grid(True, axis="y", alpha=0.3, lw=0.4)
    paper._thin_frame(ax)
    fig.savefig(fp); plt.close(fig)
    print("wrote", fp)


def ccdf(losses, sectors, fits, fp):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(**AXRECT)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for i, s in enumerate(sectors):
        x = losses.get(s)
        if x is None or len(x) < 30:
            continue
        xs = np.sort(x); n = len(xs)
        emp = 1 - np.arange(1, n + 1) / (n + 1)
        ax.plot(xs, emp, marker=markers[i % len(markers)], ms=2.6, lw=0,
                markerfacecolor="none", markeredgecolor="black",
                markeredgewidth=0.4, alpha=0.8, label=s)
        fit = fits.get(s)
        if fit and np.isfinite(fit.get("xi", np.nan)):
            u, xi, beta = fit["threshold_usd"], fit["xi"], fit["beta"]
            p_above = (xs > u).mean()
            if p_above > 0 and beta > 0 and xs.max() > u:
                xx = np.geomspace(u, xs.max(), 200)
                yy = p_above * (1.0 + xi * (xx - u) / beta) ** (-1.0 / xi)
                ax.plot(xx, yy, color="black", ls="--", lw=0.7, alpha=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    # No x-axis label: the decade ticks are in USD and the caption says so.
    ax.set_ylabel(r"Empirical CCDF  $P(X \geq L)$", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    paper._thin_frame(ax)
    ax.grid(True, alpha=0.3, which="both")
    leg = ax.legend(fontsize=FS_LEGEND, loc="lower left", ncol=2,
                    labelspacing=0.3, handletextpad=0.4, borderpad=0.4,
                    columnspacing=1.0, framealpha=0.9)
    leg.get_frame().set_linewidth(0.4)
    fig.savefig(fp); plt.close(fig)
    print("wrote", fp)


if __name__ == "__main__":
    h, sectors, losses = working_sample()
    print(f"working sample: n = {len(h)}, sectors = {sectors}")
    violin(h, sectors, OUT / "violin_slide.pdf")
    ccdf(losses, sectors, gpd_fits(losses), OUT / "ccdf_slide.pdf")
