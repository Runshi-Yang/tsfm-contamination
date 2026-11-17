"""Angle 1 on the rain-gauge pool, for models whose corpora do NOT contain these gauges.

    python -m experiments.contamination_bound.scripts.score_family --family moirai --size small

WHY THIS EXISTS. USHCN stations are in the Chronos corpus and, as far as the published lists
show, in no other family's: LOTSA (Moirai) has no USHCN or GHCN subset, Lag-Llama's checkpoint
records its 20 training sets and none is a rain-gauge archive, and TimesFM lists its public
sources (M4, electricity, traffic, Jena weather, Favorita, LibCity). For those models the direct
and indirect windows are equally out of corpus, so their direct-minus-indirect gap contains
everything that separates the two sets of gauges EXCEPT membership. That is the gap Chronos would
show if membership did nothing. Comparing the two is a difference-in-differences: it removes any
station-selection effect that the difficulty covariate misses.

Each model is scored with its own objective (see adapters.py), so gaps are compared as a
percentage of that model's indirect mean, never in raw units. The difficulty adjustment and the
station-clustered bootstrap are exactly those of adjust.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .pool import load, FILES
from .bound import metric_rng
from .adjust import difficulty, fit, boot
from . import adapters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--size", required=True)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tag = f"{a.family}-{a.size}"

    win, grp, sta, _ = load()
    keep = np.load(FILES / "keep_mask.npz")["keep"]
    win, grp, sta = win[keep].astype(np.float32), grp[keep], sta[keep]
    d, g2, _ = difficulty()
    assert len(d) == len(grp) and (g2 == grp).all(), "difficulty and pool out of sync"

    ad = adapters.REGISTRY[a.family](a.size)
    ad.eval()
    loss = ad.window_loss(win)
    np.savez(FILES / f"famscore_{tag}.npz", loss=loss, group=grp)

    sel = np.isin(grp, ["direct", "indirect"]) & np.isfinite(loss) & np.isfinite(d)
    y, ip, st, dd = loss[sel].astype(np.float64), grp[sel] == "direct", sta[sel], d[sel]
    ref = float(y[~ip].mean())
    raw = float(y[ip].mean() - ref)
    rb = boot(y, ip, np.zeros(len(y)), st, a.n_boot, metric_rng(a.seed, "fam", tag, "raw"), False)
    beta = fit(y, ip, dd)
    ab = boot(y, ip, dd, st, a.n_boot, metric_rng(a.seed, "fam", tag, "adj"), True)
    lo, hi = np.percentile(ab, [2.5, 97.5])
    res = dict(family=a.family, size=a.size, loss=ad.LOSS, ref_mean=ref,
               means={g: float(loss[grp == g].mean()) for g in ("direct", "indirect", "clean")},
               raw_diff=raw, raw_pct=100 * raw / abs(ref),
               raw_ci_pct=[100 * float(np.percentile(rb, q)) / abs(ref) for q in (2.5, 97.5)],
               adj_diff=beta, adj_pct=100 * beta / abs(ref),
               adj_ci_pct=[100 * float(lo) / abs(ref), 100 * float(hi) / abs(ref)],
               r2_with_difficulty=float(np.corrcoef(dd, y)[0, 1] ** 2))
    json.dump(res, open(FILES / f"famscore_{tag}.json", "w"), indent=2)
    print(f"[{tag}] direct - indirect: raw {res['raw_pct']:+.2f}% "
          f"[{res['raw_ci_pct'][0]:+.2f}, {res['raw_ci_pct'][1]:+.2f}], "
          f"adjusted {res['adj_pct']:+.2f}% [{res['adj_ci_pct'][0]:+.2f}, {res['adj_ci_pct'][1]:+.2f}]"
          f"  (R2 with d = {res['r2_with_difficulty']:.2f})")


if __name__ == "__main__":
    main()
