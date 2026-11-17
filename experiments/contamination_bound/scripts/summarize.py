"""Collect every checkpoint size into the tables the paper needs.

    python -m experiments.contamination_bound.scripts.summarize

Three tables:
  1. group means per metric per model size
  2. the direct-vs-indirect bound per metric per size -- the headline
  3. a metric-validity check: the fraction of windows on which two different model sizes
     produce the SAME value. A metric that answers identically for a 20M and a 709M model
     is not measuring the model, and cannot bound a model-mediated effect. This is what
     disqualified MASE and WQL here, and it is only visible because the sweep exists.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FILES = Path(__file__).resolve().parents[1] / "files"
SIZES = [("chronos-t5-mini", "20M"), ("chronos-t5-small", "46M"),
         ("chronos-t5-base", "201M"), ("chronos-t5-large", "709M")]
KEYS = [("ce_sum", "CE"), ("mase1", "MASE m=1"), ("scrps", "sCRPS"), ("wql", "WQL")]


def have(tag):
    return (FILES / f"scores_{tag}.npz").exists()


def main():
    tags = [(t, lab) for t, lab in SIZES if have(t)]
    print(f"models found: {', '.join(t for t, _ in tags)}\n")

    print("=== 1. group means ===")
    hdr = f"{'model':18}{'group':10}" + "".join(f"{lab:>12}" for _, lab in KEYS)
    print(hdr); print("-" * len(hdr))
    for tag, lab in tags:
        z = np.load(FILES / f"scores_{tag}.npz", allow_pickle=True)
        for g in ["direct", "indirect", "clean"]:
            m = z["group"] == g
            print(f"{lab + ' ' + tag.split('-')[-1]:18}{g:10}" +
                  "".join(f"{np.nanmean(z[k][m]):>12.4f}" for k, _ in KEYS))
        print()

    print("=== 2. direct vs indirect: 95% upper bound on a membership advantage ===")
    hdr = f"{'model':18}" + "".join(f"{lab:>14}" for _, lab in KEYS)
    print(hdr); print("-" * len(hdr))
    for tag, lab in tags:
        p = FILES / f"bound_{tag}.json"
        if not p.exists():
            continue
        b = json.load(open(p))
        row = []
        for k, _ in KEYS:
            r = b.get(f"direct_vs_indirect/{k}")
            row.append(f"{r['bound_pct']:>13.1f}%" if r else f"{'-':>14}")
        print(f"{lab + ' ' + tag.split('-')[-1]:18}" + "".join(row))

    print("\n=== 3. metric validity: fraction of windows scored IDENTICALLY by two sizes ===")
    print("(high = the metric does not respond to the model, so it cannot bound a "
          "model-mediated effect)")
    if len(tags) >= 2:
        a = np.load(FILES / f"scores_{tags[0][0]}.npz", allow_pickle=True)
        hdr = f"{'vs ' + tags[0][0]:26}" + "".join(f"{lab:>12}" for _, lab in KEYS)
        print(hdr); print("-" * len(hdr))
        for tag, lab in tags[1:]:
            z = np.load(FILES / f"scores_{tag}.npz", allow_pickle=True)
            row = []
            for k, _ in KEYS:
                d = np.abs(a[k] - z[k])
                d = d[np.isfinite(d)]
                row.append(f"{np.mean(d == 0) * 100:>11.1f}%")
            print(f"{lab + ' ' + tag:26}" + "".join(row))


if __name__ == "__main__":
    main()
