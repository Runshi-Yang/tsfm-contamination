"""Write the benchmark's window list for the supplementary material.

    python -m experiments.contamination_bound.scripts.export_windows <out.csv>

One row per window in the evaluation pool: GHCN station id, group, first and last date of the
576-day window, and whether it is kept (the one direct window that could not be matched value by
value against the released corpus is marked kept=0 and excluded from every result).
"""
import csv
import sys

import numpy as np

from .pool import load, FILES


def main(path):
    win, grp, sta, _ = load()
    span = np.load(FILES / "pool.npz", allow_pickle=True)["span"]
    keep = np.load(FILES / "keep_mask.npz")["keep"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["station", "group", "first_day", "last_day", "kept"])
        for s, g, sp, k in zip(sta, grp, span, keep):
            w.writerow([s, g, int(sp[0]), int(sp[1]), int(k)])
    print(path, len(sta), "windows,", int(keep.sum()), "kept")


if __name__ == "__main__":
    main(sys.argv[1])
