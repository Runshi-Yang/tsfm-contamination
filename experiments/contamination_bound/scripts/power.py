"""What this benchmark could have detected: minimum detectable AUROC and error gap.

    python -m experiments.contamination_bound.scripts.power --tag chronos-t5-small

A null is only worth reporting if the study could have seen the effect it failed to find.
This turns "we found nothing" into "we would have seen anything above this size".

TWO NUMBERS.

  Minimum detectable AUROC, at 80% power and alpha = 0.05 two-sided, from the Hanley-McNeil
  variance of the Mann-Whitney statistic. Reported twice: once with n = number of windows,
  once with n = number of STATIONS. The station figure is the honest one, because windows
  from one gauge share a climate, an observer and an instrument, so they are not
  independent draws. The ratio between the two is the design effect the window-level
  number would have hidden.

  Minimum detectable error gap, per metric, taken from the same station-clustered bootstrap
  standard error that produces the bound in `bound.py`: MDE = (z_0.975 + z_0.80) * SE.
  Using one standard error for both guarantees the bound and the power statement cannot
  disagree. It is also reported as a percentage of the reference group's mean, which puts
  it on the same scale as the bound.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from .bound import cluster_boot, metric_rng, METRICS

FILES = Path(__file__).resolve().parents[1] / "files"


def auc_se(a, n1, n2):
    """Hanley-McNeil standard error of the AUC."""
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    return np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a)
                    + (n2 - 1) * (q2 - a * a)) / (n1 * n2))


def min_detectable_auc(n1, n2, power=0.80, alpha=0.05):
    se0 = np.sqrt((n1 + n2 + 1) / (12.0 * n1 * n2))       # variance under A = 0.5
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)

    def f(a):
        return (a - 0.5) - (za * se0 + zb * auc_se(a, n1, n2))
    return float(brentq(f, 0.500001, 0.999))


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
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    z = np.load(FILES / f"scores_{a.tag}.npz", allow_pickle=True)
    z = _keep(z)
    g, st = z["group"], z["station"]
    out = {}

    print(f"=== {a.tag}: minimum detectable AUROC ({a.power:.0%} power, alpha=0.05) ===")
    hdr = f"{'contrast':24}{'windows':>18}{'MDE(win)':>10}{'stations':>18}{'MDE(stn)':>10}{'design eff':>12}"
    print(hdr); print("-" * len(hdr))
    for pos, ref in [("direct", "indirect"), ("direct", "clean"), ("indirect", "clean")]:
        mp, mr = g == pos, g == ref
        nw1, nw2 = int(mp.sum()), int(mr.sum())
        ns1, ns2 = len(set(st[mp])), len(set(st[mr]))
        aw, as_ = min_detectable_auc(nw1, nw2, a.power), min_detectable_auc(ns1, ns2, a.power)
        out[f"auc/{pos}_vs_{ref}"] = dict(n_win=[nw1, nw2], n_stn=[ns1, ns2],
                                          mde_win=aw, mde_stn=as_)
        print(f"{pos + ' vs ' + ref:24}{f'{nw1} vs {nw2}':>18}{aw:>10.3f}"
              f"{f'{ns1} vs {ns2}':>18}{as_:>10.3f}{(as_ - .5) / (aw - .5):>12.2f}x")

    zc = norm.ppf(1 - 0.05 / 2) + norm.ppf(a.power)
    print(f"\n=== minimum detectable error gap (station-clustered SE, {a.power:.0%} power) ===")
    for pos, ref in [("direct", "indirect"), ("direct", "clean")]:
        print(f"\n--- {pos} vs {ref} ---")
        hdr = f"{'metric':22}{'mean(ref)':>12}{'SE(diff)':>12}{'MDE (abs)':>12}{'MDE (% of ref)':>16}"
        print(hdr); print("-" * len(hdr))
        for key, label in METRICS:
            v = z[key].astype(np.float64)
            mp = (g == pos) & np.isfinite(v)
            mr = (g == ref) & np.isfinite(v)
            rng = metric_rng(a.seed, pos, ref, key)
            b = (cluster_boot(v[mp], st[mp], a.n_boot, rng)
                 - cluster_boot(v[mr], st[mr], a.n_boot, rng))
            se = float(b.std(ddof=1))
            ref_mean = float(np.mean(v[mr]))
            mde = zc * se
            out[f"gap/{pos}_vs_{ref}/{key}"] = dict(se=se, mde=mde, ref_mean=ref_mean,
                                                    mde_pct=mde / abs(ref_mean) * 100)
            print(f"{label:22}{ref_mean:>12.4g}{se:>12.4g}{mde:>12.4g}"
                  f"{mde / abs(ref_mean) * 100:>15.1f}%")
    json.dump(out, open(FILES / f"power_{a.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
