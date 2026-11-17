"""Is the difficulty-adjusted bound an artifact of how difficulty was defined?

    python -m experiments.contamination_bound.scripts.robust --tag chronos-t5-small

WHY THIS EXISTS. The headline bound is the coefficient on the group indicator after adjusting for
a model-free difficulty covariate. That adjustment moved the bound, so the obvious objection is
that the number depends on a definition we chose. This re-runs the same estimand under several
structurally different definitions of difficulty and reports the bound under each.

THE DEFINITIONS. The point is to vary the *structure* of the covariate, not to re-parameterise one
idea, so a reviewer cannot say we only tried variants of the same thing.

  mase0     mean|y| / seasonal-naive MAE(context) -- the MASE an identically-zero forecast
            receives. This is the definition used for the headline result: horizon magnitude
            relative to context volatility, a ratio.
  dryfrac   the share of dry days in the horizon. A count, not a ratio, and already one of the
            features the model-free gate screens.
  horizstd  the standard deviation of the horizon. Absolute spread, no reference to the context.
  ctxmae    the seasonal-naive MAE of the context alone -- the denominator of `mase0` on its own,
            so it isolates whether the ratio or its denominator is doing the work.
  none      no covariate, i.e. the unadjusted difference, for reference.

Each is computed from the target and the context only and never touches the model, so each is a
pre-treatment covariate rather than an outcome.

WHAT WOULD FALSIFY THE RESULT. If the bound is stable across these, the adjustment is not doing
anything a different reasonable choice would undo. If it moves a lot, the headline number is a
function of our definition and has to be reported as a range instead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .pool import load, FILES
from .bound import metric_rng
from .adjust import fit, boot

CTX, HOR = 512, 64


def covariates():
    """All difficulty definitions, on the value-verified windows, model-free by construction."""
    win, grp, sta, _ = load()
    p = FILES / "keep_mask.npz"
    if p.exists():
        m = np.load(p)["keep"]
        win, grp, sta = win[m], grp[m], sta[m]
    ctx = win[:, :CTX].astype(np.float64)
    hor = win[:, CTX:CTX + HOR].astype(np.float64)
    den = np.abs(ctx[:, 1:] - ctx[:, :-1]).mean(1)
    ok = den > 1e-9
    safe = np.where(ok, den, 1.0)
    return grp, sta, {
        "mase0": np.where(ok, np.abs(hor).mean(1) / safe, np.nan),
        "dryfrac": (hor <= 0).mean(1),
        "horizstd": hor.std(1),
        "ctxmae": den,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="chronos-t5-small")
    ap.add_argument("--metric", default="ce_sum")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    grp, sta, covs = covariates()
    z = np.load(FILES / f"scores_{a.tag}.npz", allow_pickle=True)
    keep = np.load(FILES / "keep_mask.npz")["keep"]
    v = z[a.metric].astype(np.float64)[keep]
    assert len(v) == len(grp), "score file and pool are out of sync"

    sel = np.isin(grp, ["direct", "indirect"]) & np.isfinite(v)
    y, ip, st = v[sel], grp[sel] == "direct", sta[sel]
    ref_mean = float(y[~ip].mean())
    print(f"=== {a.tag}, {a.metric}: bound under different difficulty definitions ===")
    print(f"reference (indirect) mean = {ref_mean:.4g}, n = {len(y)}\n")

    hdr = f"{'covariate':12}{'corr with metric':>18}{'R2':>7}{'coefficient':>13}{'95% CI':>26}{'bound':>9}"
    print(hdr); print("-" * len(hdr))
    out = {}

    # unadjusted reference
    raw = float(y[ip].mean() - y[~ip].mean())
    rb = boot(y, ip, np.zeros(len(y)), st, a.n_boot,
              metric_rng(a.seed, "none", a.metric, "raw"), False)
    lo = float(np.percentile(rb, 2.5))
    out["none"] = dict(coef=raw, lo=lo, hi=float(np.percentile(rb, 97.5)),
                       bound=max(0.0, -lo) / ref_mean * 100, r2=0.0)
    ci = "[{:.4g}, {:.4g}]".format(out["none"]["lo"], out["none"]["hi"])
    print(f"{'none (raw)':12}{'--':>18}{'--':>7}{raw:>13.4g}{ci:>26}"
          f"{out['none']['bound']:>8.1f}%")

    for name, d_all in covs.items():
        d = d_all[sel]
        good = np.isfinite(d)
        yy, ii, dd, ss = y[good], ip[good], d[good], st[good]
        r = float(np.corrcoef(dd, yy)[0, 1])
        coef = fit(yy, ii, dd)
        # Reuse adjust.py's exact random stream for the headline definition, so this table's
        # mase0 row reproduces the reported bound rather than differing by Monte-Carlo noise.
        rng = (metric_rng(a.seed, "adj", "raw", a.metric) if name == "mase0"
               else metric_rng(a.seed, "adj", name, a.metric))
        b = boot(yy, ii, dd, ss, a.n_boot, rng, True)
        lo, hi = np.percentile(b, [2.5, 97.5])
        rm = float(yy[~ii].mean())
        out[name] = dict(coef=coef, lo=float(lo), hi=float(hi),
                         bound=max(0.0, -float(lo)) / rm * 100, r2=r * r, corr=r)
        ci = "[{:.4g}, {:.4g}]".format(lo, hi)
        print(f"{name:12}{r:>18.3f}{r * r:>7.2f}{coef:>13.4g}{ci:>26}"
              f"{out[name]['bound']:>8.1f}%")

    adj = [v_["bound"] for k, v_ in out.items() if k != "none"]
    print(f"\nadjusted bounds span {min(adj):.1f}%--{max(adj):.1f}% across "
          f"{len(adj)} definitions of difficulty (unadjusted: {out['none']['bound']:.1f}%)")
    print("A covariate whose R2 with the metric is near 1 would be removing the signal along with "
          "the confound; check that column before quoting any of these.")
    json.dump(out, open(FILES / f"robust_{a.tag}_{a.metric}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
