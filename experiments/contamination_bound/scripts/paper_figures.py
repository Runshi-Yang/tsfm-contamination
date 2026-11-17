"""Draw the paper's two result figures from the saved results.

    python -m experiments.contamination_bound.scripts.paper_figures <overleaf>/figures
    python -m experiments.contamination_bound.scripts.paper_figures <dir> --poster   # poster PNGs

step1.pdf   direct-minus-indirect loss for all nine checkpoints, difficulty-adjusted, in standard
            deviations of each model's loss (files/famcompare.json), with the Chronos-minus-others row.
family.pdf  share of planted cores memorised against exposures, exact copies and blends, for the six
            planted checkpoints (files/family_*.json). Each panel names its own checkpoints.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .pool import FILES

BLUE, LIGHT, ORANGE = "#2a78d6", "#86b6ef", "#eb6834"
INK, MUTED, GRID, SURF, GREY = "#0b0b0b", "#52514e", "#dcdcd8", "#ffffff", "#8a8983"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.facecolor": SURF,
                     "figure.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})


def step1(out, poster=False):
    d = json.load(open(FILES / "famcompare.json"))
    heads = (("Models trained on these rain gauges", "Models never trained on these rain gauges") if poster
             else ("Corpus contains these gauges", "Published corpus does not list them"))
    rows = ([("head", heads[0], None)]
            + [("data", n, t) for t, n in [("chronos-mini", "Chronos 20M"), ("chronos-small", "Chronos 46M"),
                                            ("chronos-base", "Chronos 201M"), ("chronos-large", "Chronos 709M")]]
            + [("head", heads[1], None)]
            + [("data", n, t) for t, n in [("moirai-small", "Moirai small"), ("moirai-large", "Moirai large"),
                                            ("timesfm-200m", "TimesFM 1.0-200M"), ("timesfm-500m", "TimesFM 2.0-500M"),
                                            ("lagllama-base", "Lag-Llama")]]
            + ([] if poster else [("head", "", None), ("did", "Chronos minus the others", "did")]))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = np.arange(len(rows))[::-1]
    x0 = -0.62
    for yy, (k, lab, t) in zip(y, rows):
        if k == "head":
            if lab:
                ax.text(x0 + 0.01, yy, lab, color=INK, fontsize=10, fontweight="600", va="center")
            continue
        ax.text(x0 + 0.04, yy, lab, color=INK if k == "did" else MUTED, fontsize=9.5, va="center",
                fontweight="600" if k == "did" else "normal")
        if k == "did":
            c, e, (lo, hi) = ORANGE, d["did"]["est_sd"], d["did"]["ci_sd"]
        else:
            c = BLUE if d[t]["in_corpus"] else GREY
            e, (lo, hi) = d[t]["adj"], d[t]["adj_ci"]
        ax.plot([lo, hi], [yy, yy], color=c, lw=2.3, solid_capstyle="butt", zorder=2)
        for x in (lo, hi):
            ax.plot([x, x], [yy - .18, yy + .18], color=c, lw=2.3, zorder=2)
        ax.plot(e, yy, "o", ms=7, color=c, mec=SURF, mew=1.5, zorder=3)
    ax.axvline(0, color=INK, lw=1.2, zorder=1)
    ax.set_xlim(x0, 0.34); ax.set_ylim(-0.7, len(rows) - 0.4)
    ax.set_xticks([-0.1, 0, 0.1, 0.2, 0.3]); ax.set_yticks([]); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=.8); ax.set_axisbelow(True)
    ax.set_xlabel("standardized loss difference (direct \u2212 indirect)" if poster
                  else "standardized loss difference, direct $-$ indirect, adjusted for difficulty")
    fig.tight_layout()
    fig.savefig(out / ("poster_fig1.png" if poster else "step1.pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def family(out, poster=False):
    FAM = [("Chronos", [("chronos-small", "46M"), ("chronos-large", "709M")], "center right", None),
           ("Moirai", [("moirai-small", "small")], "center right", None),
           ("TimesFM", [("timesfm-200m-lr1e-05-perepoch", "1.0-200M"), ("timesfm-500m-lr1e-05", "2.0-500M")],
            "lower right", (1.0, 0.18)),
           ("Lag-Llama", [("lagllama-base", "one size")], "upper left", None)]
    KS = [1, 2, 4, 8, 16, 32]
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 4.3), sharex=True, sharey=True)
    for col, (name, runs, loc, anchor) in enumerate(FAM):
        for j, (tag, lab) in enumerate(runs):
            r = json.load(open(FILES / f"family_{tag}.json"))
            c = BLUE if (j == len(runs) - 1) else LIGHT
            for row, form in enumerate(["verbatim", "mixup"]):
                cells = [r["cells"][f"{form}/k{k}"] for k in KS]
                axes[row, col].plot([cc["exposures"] for cc in cells], [cc["frac"] for cc in cells], "-o",
                                    color=c, lw=1.8, ms=5, mec=SURF, mew=1.2, zorder=3,
                                    label=lab if row == 0 else None)
        axes[0, col].set_title(name, color=INK, fontsize=10, fontweight="600", pad=6)
        axes[0, col].legend(frameon=False, fontsize=8.5, labelcolor=MUTED, loc=loc, handlelength=1.4,
                            borderaxespad=0.2, bbox_to_anchor=anchor)
    for a in axes.ravel():
        a.axhline(0.05, color=MUTED, lw=0.9, ls=(0, (2, 2)), zorder=1)
        a.set_xscale("log"); a.set_xlim(7, 460); a.set_ylim(-0.07, 1.08)
        a.set_xticks([10, 40, 160]); a.set_xticklabels(["10", "40", "160"]); a.minorticks_off()
        a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
    axes[0, 0].set_ylabel("direct\n(exact copies)", color=INK, fontsize=9.5, fontweight="600")
    axes[1, 0].set_ylabel("indirect\n(blends)", color=INK, fontsize=9.5, fontweight="600")
    fig.supxlabel("times the model saw the window during training", color=MUTED, fontsize=9.5, y=0.035)
    fig.supylabel("share memorised", color=MUTED, fontsize=9.5, x=0.005)
    fig.tight_layout(rect=[0.02, 0.04, 1, 1])
    fig.savefig(out / ("poster_fig2.png" if poster else "family.pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    poster = "--poster" in sys.argv[2:]          # PNG versions: no DiD row, plain labels
    step1(out, poster); family(out, poster)
    print("figures written to", out)
