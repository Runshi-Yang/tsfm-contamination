"""Put the direct-minus-indirect gap of every model on one scale.

    python -m experiments.contamination_bound.scripts.famcompare

WHY NOT PERCENT OF THE MEAN. For Chronos the loss is a cross-entropy, which is non-negative and has
a natural zero, so a gap as a percentage of the mean is meaningful and is what the paper quotes.
Moirai and Lag-Llama are scored by the negative log-likelihood of a continuous density, whose zero
depends on the units of the data; Lag-Llama's mean NLL on this pool is close to zero, and dividing
by it produces gaps of +80% that mean nothing. So for the comparison ACROSS models we standardise:
each model's per-window loss is divided by its standard deviation over the direct and indirect
windows, and the gap is reported in those units (a standardised mean difference). Raw and
difficulty-adjusted, with the same station bootstrap as everywhere else.

WHAT TO LOOK FOR. Models whose corpora contain no USHCN gauges have no membership to benefit from,
so their gap is what the gauges themselves contribute. If Chronos's gap is no more negative than
theirs, membership bought Chronos nothing beyond what the gauges explain.
"""
from __future__ import annotations

import json

import numpy as np

from .pool import load, FILES
from .bound import metric_rng
from .adjust import difficulty, fit, boot

TAGS = ["chronos-mini", "chronos-small", "chronos-base", "chronos-large", "moirai-small", "moirai-large",
        "timesfm-200m", "timesfm-500m", "lagllama-base"]
IN_CORPUS = {"chronos"}          # families whose published corpus lists USHCN


def main(n_boot=10_000, seed=0):
    _, grp, sta, _ = load()
    keep = np.load(FILES / "keep_mask.npz")["keep"]
    grp, sta = grp[keep], sta[keep]
    d, _, _ = difficulty()
    out = {}
    print(f"{'model':15}{'USHCN in corpus':>16}{'raw gap (sd)':>26}{'adjusted gap (sd)':>28}")
    for tag in TAGS:
        p = FILES / f"famscore_{tag}.npz"
        if not p.exists():
            continue
        loss = np.load(p)["loss"].astype(np.float64)
        sel = np.isin(grp, ["direct", "indirect"]) & np.isfinite(loss) & np.isfinite(d)
        y, ip, st, dd = loss[sel], grp[sel] == "direct", sta[sel], d[sel]
        y = (y - y.mean()) / y.std()
        raw = float(y[ip].mean() - y[~ip].mean())
        rb = boot(y, ip, np.zeros(len(y)), st, n_boot, metric_rng(seed, "cmp", tag, "raw"), False)
        adj = fit(y, ip, dd)
        ab = boot(y, ip, dd, st, n_boot, metric_rng(seed, "cmp", tag, "adj"), True)
        member = tag.split("-")[0] in IN_CORPUS
        out[tag] = dict(in_corpus=member, raw=raw, raw_ci=list(map(float, np.percentile(rb, [2.5, 97.5]))),
                        adj=adj, adj_ci=list(map(float, np.percentile(ab, [2.5, 97.5]))),
                        r2_d=float(np.corrcoef(dd, y)[0, 1] ** 2))
        o = out[tag]
        print(f"{tag:15}{'yes' if member else 'no':>16}"
              f"{o['raw']:>+9.3f} [{o['raw_ci'][0]:+.3f}, {o['raw_ci'][1]:+.3f}]"
              f"{o['adj']:>+11.3f} [{o['adj_ci'][0]:+.3f}, {o['adj_ci'][1]:+.3f}]")
    # Difference in differences. All models are scored on the same windows, so one station
    # resample is drawn per bootstrap draw and every model's adjusted gap is computed on it;
    # the interval then accounts for the models' gaps being correlated through shared windows.
    z = {t: np.load(FILES / f"famscore_{t}.npz")["loss"].astype(np.float64) for t in out}
    sel = np.isin(grp, ["direct", "indirect"]) & np.isfinite(d)
    for t in z:
        sel &= np.isfinite(z[t])
    ip, st, dd = grp[sel] == "direct", sta[sel], d[sel]
    Y = {t: (z[t][sel] - z[t][sel].mean()) / z[t][sel].std() for t in z}
    mem = [t for t in Y if out[t]["in_corpus"]]
    non = [t for t in Y if not out[t]["in_corpus"]]
    est = np.mean([fit(Y[t], ip, dd) for t in mem]) - np.mean([fit(Y[t], ip, dd) for t in non])
    gap = lambda y, m: y[m].mean() - y[~m].mean()
    est_raw = np.mean([gap(Y[t], ip) for t in mem]) - np.mean([gap(Y[t], ip) for t in non])
    uniq, inv = np.unique(st, return_inverse=True)
    by = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = metric_rng(seed, "cmp", "did", "adj")
    draws, draws_raw = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([by[p] for p in rng.integers(0, len(by), len(by))])
        g = {t: fit(Y[t][idx], ip[idx], dd[idx]) for t in Y}
        draws[b] = np.mean([g[t] for t in mem]) - np.mean([g[t] for t in non])
        r = {t: gap(Y[t][idx], ip[idx]) for t in Y}          # same resample, no covariate
        draws_raw[b] = np.mean([r[t] for t in mem]) - np.mean([r[t] for t in non])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    rlo, rhi = np.percentile(draws_raw, [2.5, 97.5])
    # back to Chronos's own units: sd of CE over the pool, as a share of the indirect mean
    ce = z["chronos-small"][sel]
    to_pct = 100 * ce.std() / ce[~ip].mean()
    out["did"] = dict(members=mem, non_members=non, est_sd=float(est), ci_sd=[float(lo), float(hi)],
                      pct_of_ce_46m=[float(est * to_pct), float(lo * to_pct), float(hi * to_pct)],
                      bound_pct_46m=float(max(0.0, -lo) * to_pct),
                      # the raw contrast shares no covariate with the main bound: the independent check
                      raw_est_sd=float(est_raw), raw_ci_sd=[float(rlo), float(rhi)],
                      raw_pct_of_ce_46m=[float(est_raw * to_pct), float(rlo * to_pct), float(rhi * to_pct)],
                      raw_bound_pct_46m=float(max(0.0, -rlo) * to_pct))
    print(f"difference in differences, RAW: {est_raw:+.3f} sd [{rlo:+.3f}, {rhi:+.3f}] = "
          f"{est_raw * to_pct:+.2f}% [{rlo * to_pct:+.2f}, {rhi * to_pct:+.2f}] -> bound {max(0.0, -rlo) * to_pct:.2f}%")
    print(f"\ndifference in differences (members minus non-members, adjusted): {est:+.3f} sd "
          f"[{lo:+.3f}, {hi:+.3f}]  = {est * to_pct:+.2f}% of 46M CE [{lo * to_pct:+.2f}, {hi * to_pct:+.2f}]"
          f"  -> bound {max(0.0, -lo) * to_pct:.2f}%")
    json.dump(out, open(FILES / "famcompare.json", "w"), indent=2)


if __name__ == "__main__":
    main()
