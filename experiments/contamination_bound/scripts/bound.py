"""Confidence interval on the direct-vs-indirect error gap, and the relative bound.

    python -m experiments.contamination_bound.scripts.bound --tag chronos-t5-small

For each metric: the mean error of each group, the difference (direct minus indirect,
where NEGATIVE means direct is easier for the model, which is the shape leakage would
have), a bootstrap interval on that difference, and the interval's most negative end
expressed as a percentage of the reference group's mean error. That percentage is the
largest forecasting advantage training-set membership could still have and be consistent
with this data.

RESAMPLING UNIT IS THE STATION, NOT THE WINDOW. Windows from one gauge are not
independent -- they share a climate, an observer and an instrument -- so resampling
windows would treat correlated observations as fresh evidence and give an interval that
is too narrow. The window-level interval is printed beside it only to show the size of
that effect, and should not be the number that gets quoted.

WHAT THE BOUND COVERS. direct and indirect are both US co-op precipitation gauges in the
same years, so whatever the model absorbed about US weather in general is present on both
sides and cancels in the difference. What the contrast isolates is the increment between
their measured correlation with contemporaneous corpus data (printed below as
`corpus_corr`). The bound therefore covers exact membership plus the top slice of
realisation-level correlation -- not domain familiarity, which cannot be bounded this way
because any comparable control data came from the same public pools the model trained on.

The direct-vs-clean contrast is reported too, but it varies period as well as membership,
so it mixes leakage with two years of drift and is the weaker of the two.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from zlib import crc32

FILES = Path(__file__).resolve().parents[1] / "files"
METRICS = [("ce_sum", "CE (summed tokens)"), ("ce_mean", "CE (per token)"),
           ("mase1", "MASE m=1"), ("mase7", "MASE m=7"),
           ("scrps", "scaled CRPS"), ("wql", "WQL (degenerate here)")]


def cluster_boot(vals, stations, n_boot, rng):
    """Bootstrap the mean by resampling stations with replacement."""
    uniq, inv = np.unique(stations, return_inverse=True)
    by = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(by), len(by))
        idx = np.concatenate([by[p] for p in pick])
        out[b] = np.nanmean(vals[idx])
    return out


def window_boot(vals, n_boot, rng):
    idx = rng.integers(0, len(vals), (n_boot, len(vals)))
    return np.nanmean(vals[idx], axis=1)


def metric_rng(seed, pos, ref, key):
    """An independent, deterministic generator per (contrast, metric).

    Drawing every metric from one shared generator makes each metric's interval depend on
    how many metrics were computed before it, so adding a metric to METRICS silently moves
    the numbers for all the others. Deriving a separate stream per cell keeps each interval
    reproducible on its own."""
    tag = f"{pos}|{ref}|{key}".encode()
    return np.random.default_rng([seed, crc32(tag)])


def contrast(z, key, pos, ref, n_boot, seed):
    rng = metric_rng(seed, pos, ref, key)
    g, st = z["group"], z["station"]
    v = z[key].astype(np.float64)
    mp, mr = (g == pos) & np.isfinite(v), (g == ref) & np.isfinite(v)
    vp, vr, sp, sr = v[mp], v[mr], st[mp], st[mr]

    diff = float(np.mean(vp) - np.mean(vr))
    bc = cluster_boot(vp, sp, n_boot, rng) - cluster_boot(vr, sr, n_boot, rng)
    bw = window_boot(vp, n_boot, rng) - window_boot(vr, n_boot, rng)
    lo, hi = np.percentile(bc, [2.5, 97.5])
    wlo, whi = np.percentile(bw, [2.5, 97.5])
    ref_mean = float(np.mean(vr))
    bound = max(0.0, -lo) / abs(ref_mean) * 100.0
    return dict(n_pos=int(mp.sum()), n_ref=int(mr.sum()),
                st_pos=int(len(set(sp))), st_ref=int(len(set(sr))),
                mean_pos=float(np.mean(vp)), mean_ref=ref_mean, diff=diff,
                lo=float(lo), hi=float(hi), w_lo=float(wlo), w_hi=float(whi),
                bound_pct=float(bound))


def _keep(z):
    """Restrict to windows whose membership was verified at the value level.

    `direct` membership was established from station identifiers, which says the station is in
    the corpus, not that our exact values are among the released rows. verify_direct.py checks
    each window against the released file day by day; anything it could not confirm is dropped
    here so that every number below is computed on data the model demonstrably saw."""
    p = FILES / "keep_mask.npz"
    if not p.exists():
        return {k: z[k] for k in z.files}
    m = np.load(p)["keep"]
    return {k: (z[k][m] if getattr(z[k], "shape", (0,)) and len(z[k]) == len(m) else z[k])
            for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="chronos-t5-small")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    z = np.load(FILES / f"scores_{a.tag}.npz", allow_pickle=True)
    z = _keep(z)

    g, corr = z["group"], z["corpus_corr"]
    print(f"=== {a.tag} ===")
    print("measured correlation with contemporaneous corpus data (p50):")
    for grp in ["direct", "indirect", "clean"]:
        m = g == grp
        print(f"  {grp:9} n={int(m.sum()):5} stations={len(set(z['station'][m])):5} "
              f"corpus_corr={np.median(corr[m]):.3f}")

    out = {}
    for pos, ref in [("direct", "indirect"), ("direct", "clean")]:
        print(f"\n--- {pos} vs {ref} "
              f"{'(varies membership only)' if ref == 'indirect' else '(also varies period)'} ---")
        hdr = (f"{'metric':22}{'mean(pos)':>11}{'mean(ref)':>11}{'diff':>10}"
               f"{'95% CI (station)':>26}{'  (window)':>22}{'bound':>9}")
        print(hdr); print("-" * len(hdr))
        for key, label in METRICS:
            r = contrast(z, key, pos, ref, a.n_boot, a.seed)
            out[f"{pos}_vs_{ref}/{key}"] = r
            ci_st = "[{:.4g}, {:.4g}]".format(r["lo"], r["hi"])
            ci_wn = "[{:.4g}, {:.4g}]".format(r["w_lo"], r["w_hi"])
            print(f"{label:22}{r['mean_pos']:>11.4g}{r['mean_ref']:>11.4g}"
                  f"{r['diff']:>10.4g}{ci_st:>26}{ci_wn:>22}"
                  f"{r['bound_pct']:>8.1f}%")
        print("\nsentences:")
        for key, label in METRICS:
            r = out[f"{pos}_vs_{ref}/{key}"]
            print(f"  [{label}] difference {r['diff']:.4g}, 95% CI "
                  f"[{r['lo']:.4g}, {r['hi']:.4g}], reference mean {r['mean_ref']:.4g} "
                  f"-> any advantage is below {r['bound_pct']:.1f}%")
    json.dump(out, open(FILES / f"bound_{a.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
