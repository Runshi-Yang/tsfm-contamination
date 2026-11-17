"""Sensitivity of the bound to the resampling unit: stations versus geographic blocks.

    python -m experiments.contamination_bound.scripts.blockboot

WHY. Nearby gauges record the same storms (the median indirect station is 39 km from the nearest
corpus station), so resampling stations treats correlated gauges as independent and can make the
interval too narrow. Here the same estimands -- the raw gap and the difficulty-adjusted coefficient,
direct vs indirect, token CE -- are bootstrapped with whole lat/lon cells as the resampling unit, at
two cell sizes. Cells take every window of every station inside them, so gauges that share weather
are drawn together.
"""
from __future__ import annotations

import json

import numpy as np

from .pool import load, FILES, inventory
from .bound import metric_rng
from .adjust import difficulty, fit

SIZES = ["mini", "small", "base", "large"]
SCHEMES = {"station": None, "2deg": 2.0, "5deg": 5.0}


def blocks(sta, deg):
    if deg is None:
        return sta
    inv = inventory()
    ll = np.array([inv[s][:2] for s in sta])
    return np.array([f"{int(np.floor(a / deg))}_{int(np.floor(b / deg))}" for a, b in ll])


def boot(y, ip, d, unit, n_boot, rng, adjusted):
    uniq, inv = np.unique(unit, return_inverse=True)
    by = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([by[p] for p in rng.integers(0, len(by), len(by))])
        if adjusted:
            out[b] = fit(y[idx], ip[idx], d[idx])
        else:
            out[b] = y[idx][ip[idx]].mean() - y[idx][~ip[idx]].mean()
    return out, len(uniq)


def main(n_boot=10_000, seed=0):
    _, grp, sta, _ = load()
    keep = np.load(FILES / "keep_mask.npz")["keep"]
    grp, sta = grp[keep], sta[keep]
    d, _, _ = difficulty()
    sel = np.isin(grp, ["direct", "indirect"]) & np.isfinite(d)
    out = {}
    for s in SIZES:
        z = np.load(FILES / f"scores_chronos-t5-{s}.npz", allow_pickle=True)
        y = z["ce_sum"][keep][sel].astype(np.float64)
        ip, st, dd = grp[sel] == "direct", sta[sel], d[sel]
        ref = y[~ip].mean()
        for name, deg in SCHEMES.items():
            unit = blocks(st, deg)
            for adj in (False, True):
                rng = metric_rng(seed, "block", f"{s}-{name}", "adj" if adj else "raw")
                b, nu = boot(y, ip, dd, unit, n_boot, rng, adj)
                lo, hi = np.percentile(b, [2.5, 97.5])
                key = f"{s}/{name}/{'adj' if adj else 'raw'}"
                out[key] = dict(lo=float(lo), hi=float(hi), bound=float(max(0, -lo) / ref * 100), n_units=int(nu))
                print(f"{key:22} units {nu:4d}  [{lo:7.2f}, {hi:6.2f}]  bound {out[key]['bound']:.1f}%")
    json.dump(out, open(FILES / "blockboot.json", "w"), indent=2)


if __name__ == "__main__":
    main()
