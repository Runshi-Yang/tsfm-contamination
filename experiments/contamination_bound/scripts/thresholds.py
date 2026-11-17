"""Smallest detectable dose per checkpoint and form, the paper's threshold table.

    python -m experiments.contamination_bound.scripts.thresholds

Reads the per-cell results that run_family.py wrote and applies both tests of the paper to every
cell, Bonferroni-corrected over the cells of the grid being read:
  share (binom)  the number of the ten cores above the controls' 95th percentile, against
                 Binomial(10, 0.05); four or more passes over the 18-cell main grid (p = 0.001),
                 five or more over the 180-cell per-epoch grid;
  mean (welch)   run_family's two-sided Welch test of the cores against the controls, with a
                 positive excess.
A threshold is the smallest dose, in exposures, at which a cell passes ("-" if none does).

SOURCES. The main grid of each checkpoint is its family_<tag>.json at the learning rate chosen by
lr_check.py. TimesFM 200M has no separate main-grid file at that rate: its per-epoch rerun used the
same seed and wrote over it, and its cells after the tenth epoch are that run. The per-epoch grids
score every window after each epoch, so a core planted k times is read at k, 2k, ..., 10k exposures.
"""
from __future__ import annotations

import json

from scipy import stats

from .inject import KS, TYPES
from .pool import FILES

MAIN = {"chronos-small": "chronos-small", "chronos-large": "chronos-large",
        "timesfm-200m": "timesfm-200m-lr1e-05-perepoch", "timesfm-500m": "timesfm-500m-lr1e-05",
        "moirai-small": "moirai-small", "lagllama": "lagllama-base"}
PER_EPOCH = {"chronos-large (per epoch)": "chronos-large-perepoch",
             "timesfm-200m (per epoch)": "timesfm-200m-lr1e-05-perepoch"}
N_CORES, CHANCE = 10, 0.05


def first_dose(cells, alpha):
    """cells: (form, exposures, cell dict) triples. Returns the smallest passing dose per test."""
    need = next(c for c in range(N_CORES + 1) if stats.binom.sf(c - 1, N_CORES, CHANCE) < alpha)
    out = {"welch": {}, "binom": {}}
    for t in TYPES:
        own = sorted(((e, c) for form, e, c in cells if form == t), key=lambda x: x[0])
        welch = [e for e, c in own if c["p"] < alpha and c["excess"] > 0]
        binom = [e for e, c in own if round(N_CORES * c["frac"]) >= need]
        out["welch"][t] = welch[0] if welch else "-"
        out["binom"][t] = binom[0] if binom else "-"
    out["alpha"] = alpha
    return out


def main():
    res = {}
    for name, tag in MAIN.items():
        d = json.load(open(FILES / f"family_{tag}.json"))
        cells = [(t, k * d["epochs"], d["cells"][f"{t}/k{k}"]) for t in TYPES for k in KS]
        res[name] = first_dose(cells, 0.05 / len(cells))
    for name, tag in PER_EPOCH.items():
        d = json.load(open(FILES / f"family_{tag}.json"))
        cells = [(t, k * int(e), r["cells"][f"{t}/k{k}"])
                 for e, r in d["per_epoch"].items() for t in TYPES for k in KS]
        res[name] = first_dose(cells, 0.05 / len(cells))
    json.dump(res, open(FILES / "family_thresholds.json", "w"))
    for name, r in res.items():
        print(f"{name:26}" + "  ".join(f"{t} {r['binom'][t]} ({r['welch'][t]})" for t in TYPES))


if __name__ == "__main__":
    main()
