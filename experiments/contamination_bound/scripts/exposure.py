"""The exposure margin: at what dose does injection become readable, and how far below is it.

    python -m experiments.contamination_bound.scripts.exposure --epochs 10

Reads the injection run and answers the question Step 4 asks: how far below the exposure at
which memorisation becomes measurable does a released checkpoint sit.

THE UNIT IS TOTAL EXPOSURES, NOT k. A core window placed in the corpus k times and trained for
E epochs is presented to the model k*E times, not k. The plan's phrasing ("memorisation is
visible around k = 8, Chronos sees each window once, so the checkpoint is 8x below") treats k
and exposure count as the same thing; they differ by a factor of E. Both are reported below,
and the margin is computed on total exposures.

WHAT COUNTS AS READABLE. delta(z) = CE(released) - CE(contaminated) for each window: positive
means the contaminated model fits z better. The held-out clean windows give the null spread of
delta -- they were never injected, but they still move because the model was fine-tuned. A cell
is called readable when its mean delta exceeds the clean mean by more than the clean spread can
account for, judged by a two-sided t-test on the cell's ten cores against the clean baseline at
the 5% level, and additionally by the fraction of the cell's cores that exceed the clean 95th
percentile. Both are reported, because a mean can be moved by one core.

WHY THE ANSWER IS A LOWER BOUND. Injected copies are 19% of this corpus, against roughly one
window in ten million for real pretraining, and the fine-tune runs many epochs over a small
corpus. Both make memorisation easier to produce than it would be at pretraining scale, so the
exposure this experiment identifies is a floor: a realistically diluted corpus would need more.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from .inject import OUT, KS, TYPES

FILES = Path(__file__).resolve().parents[1] / "files"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    a = ap.parse_args()

    z = np.load(OUT / f"inject_delta_ep{a.epochs}.npz", allow_pickle=True)
    delta, nc = z["delta"], int(z["n_core"])
    ctype, ck = z["core_type"], z["core_k"]
    d_core, d_clean = delta[:nc], delta[nc:]
    base, spread = float(d_clean.mean()), float(d_clean.std(ddof=1))
    p95 = float(np.percentile(d_clean, 95))

    print(f"=== injection run, {a.epochs} epochs ===")
    print(f"clean baseline delta: mean {base:+.3f}, sd {spread:.3f}, 95th pct {p95:+.3f}  "
          f"(n={len(d_clean)})")
    print("\nExcess delta over the clean baseline, in summed nats. A cell is 10 core windows.")
    print("'frac' is the share of the cell's cores above the clean 95th percentile "
          "(0.05 expected under no effect).\n")

    hdr = f"{'type':10}{'k':>4}{'exposures':>11}{'excess':>10}{'p':>9}{'frac':>7}   readable"
    print(hdr); print("-" * len(hdr))
    res, first = {}, {}
    for t in TYPES:
        for k in KS:
            m = (ctype == t) & (ck == k)
            if not m.any():
                continue
            v = d_core[m]
            excess = float(v.mean() - base)
            p = float(stats.ttest_ind(v, d_clean, equal_var=False).pvalue)
            frac = float((v > p95).mean())
            ok = (p < 0.05) and (excess > 0)
            res[f"{t}/k{k}"] = dict(excess=excess, p=p, frac=frac, readable=bool(ok),
                                    exposures=int(k) * a.epochs)
            if ok and t not in first:
                first[t] = int(k)
            print(f"{t:10}{k:>4}{k * a.epochs:>11}{excess:>10.2f}{p:>9.3g}{frac:>7.2f}"
                  f"   {'yes' if ok else 'no'}")

    print("\n=== the margin ===")
    for t in TYPES:
        if t in first:
            k = first[t]
            print(f"  {t:9} first readable at k={k}, i.e. {k * a.epochs} total exposures "
                  f"-> a released checkpoint at ~1 exposure sits {k * a.epochs}x below it")
        else:
            print(f"  {t:9} never readable at any k up to {max(KS)} "
                  f"({max(KS) * a.epochs} exposures): no margin can be quoted, and the honest "
                  f"statement is that this contamination type left no measurable trace")
    json.dump({"epochs": a.epochs, "clean_mean": base, "clean_sd": spread,
               "clean_p95": p95, "first_readable_k": first, "cells": res},
              open(FILES / f"exposure_ep{a.epochs}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
