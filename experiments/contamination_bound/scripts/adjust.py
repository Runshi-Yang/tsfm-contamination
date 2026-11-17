"""The bound after adjusting for intrinsic window difficulty.

    python -m experiments.contamination_bound.scripts.adjust --tag chronos-t5-small

WHY THIS EXISTS. Step 1 showed that MASE is essentially model-free on this data: a 20M and a
709M checkpoint score 97-99% of windows identically, because ~72% of days are dry, the sample
median is zero almost everywhere, and the point forecast collapses to the all-zero forecast.
That makes MASE a measurement of how hard a window is, not of how well a model does.

And measured that way, the `direct` group is 5.2% HARDER than `indirect` -- larger than the
3.7% cross-entropy gap we are trying to bound. So the two groups are not exchangeable on
difficulty, and the raw contrast attributes to membership something that is partly a property
of the windows. The model-free gate did not catch this: it screens single hand-crafted
spectral and roughness features, and difficulty in this sense is not one of them.

THE COVARIATE. d(z) = mean|y| / mean|Y_t - Y_{t-1}|, i.e. the MASE that an identically-zero
forecast would receive. It is computed from the target and the context only and never touches
the model, so it is a pre-treatment covariate rather than an outcome.

THE ADJUSTED ESTIMAND. The coefficient on the group indicator in

    metric ~ 1 + 1[direct] + d

with a station-clustered bootstrap over the same resampling scheme used for the raw bound.
Raw and adjusted are always reported together: the adjustment moves the estimate, and hiding
either number would misrepresent how much of the raw gap was difficulty.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .pool import load
from .bound import metric_rng

FILES = Path(__file__).resolve().parents[1] / "files"
CTX, HOR = 512, 64
METRICS = [("ce_sum", "CE (summed tokens)"), ("scrps", "scaled CRPS")]


def difficulty():
    """d(z): the MASE an identically-zero forecast receives. Model-free by construction.

    Applies the same value-level verification mask as the score files, so the covariate, the
    groups and the metrics all describe the same set of windows. Without this the arrays differ
    in length and the consistency assert in main() fires."""
    win, grp, sta, _ = load()
    p = FILES / "keep_mask.npz"
    if p.exists():
        m = np.load(p)["keep"]
        win, grp, sta = win[m], grp[m], sta[m]
    ctx = win[:, :CTX].astype(np.float64)
    hor = win[:, CTX:CTX + HOR].astype(np.float64)
    den = np.abs(ctx[:, 1:] - ctx[:, :-1]).mean(1)
    ok = den > 1e-9
    return np.where(ok, np.abs(hor).mean(1) / np.where(ok, den, 1), np.nan), grp, sta


def fit(y, is_pos, d):
    """OLS coefficient on the group indicator, adjusting for difficulty."""
    X = np.column_stack([np.ones(len(y)), is_pos.astype(float), d])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1])


def boot(y, is_pos, d, st, n_boot, rng, adjusted):
    uniq, inv = np.unique(st, return_inverse=True)
    by = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([by[p] for p in rng.integers(0, len(by), len(by))])
        if adjusted:
            out[b] = fit(y[idx], is_pos[idx], d[idx])
        else:
            p, q = is_pos[idx], ~is_pos[idx]
            out[b] = y[idx][p].mean() - y[idx][q].mean()
    return out


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

    d_all, grp, sta = difficulty()
    z = np.load(FILES / f"scores_{a.tag}.npz", allow_pickle=True)
    z = _keep(z)
    assert (z["group"] == grp).all(), "score file and pool are out of sync"

    print(f"=== {a.tag} ===")
    print("model-free difficulty d = mean|y| / seasonal-naive MAE(context)")
    for g in ["direct", "indirect", "clean"]:
        m = grp == g
        print(f"  {g:9} d = {np.nanmean(d_all[m]):.4f}")
    dd = np.nanmean(d_all[grp == "direct"]) - np.nanmean(d_all[grp == "indirect"])
    print(f"  direct - indirect = {dd:+.4f}  "
          f"({dd / np.nanmean(d_all[grp == 'indirect']) * 100:+.1f}% of indirect)\n")

    out = {}
    hdr = (f"{'metric':22}{'raw diff':>10}{'raw bound':>11}"
           f"{'adj diff':>10}{'adj 95% CI':>26}{'adj bound':>11}{'R2 w/ d':>9}")
    print(hdr); print("-" * len(hdr))
    for key, label in METRICS:
        v = z[key].astype(np.float64)
        m = np.isin(grp, ["direct", "indirect"]) & np.isfinite(v) & np.isfinite(d_all)
        y, ip, d, st = v[m], grp[m] == "direct", d_all[m], sta[m]
        ref_mean = float(y[~ip].mean())

        # How much of this metric IS the covariate? If d explains nearly all of it, the
        # adjustment removes the signal along with the confound and the adjusted bound is
        # an artifact of collinearity, not a tighter result. Reported so that cannot hide.
        r2 = float(np.corrcoef(d, y)[0, 1] ** 2)

        # Read the unadjusted bound from bound.py rather than recomputing it here. An
        # independent recomputation lands a tenth of a point away through Monte Carlo alone,
        # which would put two different "raw bound" numbers for the same cell on record.
        raw = float(y[ip].mean() - y[~ip].mean())
        raw_bound = json.load(open(FILES / f"bound_{a.tag}.json"))[
            f"direct_vs_indirect/{key}"]["bound_pct"]

        adj = fit(y, ip, d)
        ab = boot(y, ip, d, st, a.n_boot, metric_rng(a.seed, "adj", "raw", key), True)
        lo, hi = np.percentile(ab, [2.5, 97.5])

        r = dict(raw_diff=raw, raw_bound=raw_bound,
                 adj_diff=adj, adj_lo=float(lo), adj_hi=float(hi),
                 adj_bound=max(0.0, -float(lo)) / ref_mean * 100, ref_mean=ref_mean,
                 r2_with_difficulty=r2)
        out[key] = r
        ci = "[{:.4g}, {:.4g}]".format(lo, hi)
        print(f"{label:22}{raw:>10.4g}{r['raw_bound']:>10.1f}%"
              f"{adj:>10.4g}{ci:>26}{r['adj_bound']:>10.1f}%{r2:>9.3f}")

    print("\nsentences (adjusted is the primary result):")
    for key, label in METRICS:
        r = out[key]
        print(f"  [{label}] after adjusting for difficulty the gap is {r['adj_diff']:.4g} "
              f"(95% CI [{r['adj_lo']:.4g}, {r['adj_hi']:.4g}]); any membership advantage is "
              f"below {r['adj_bound']:.1f}% of the reference mean {r['ref_mean']:.4g} "
              f"(unadjusted: {r['raw_bound']:.1f}%).")
    json.dump(out, open(FILES / f"adjust_{a.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
