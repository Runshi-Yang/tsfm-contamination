"""Verify the `direct` group at the value level, not the station level.

    python -m experiments.contamination_bound.scripts.verify_direct

WHY. Membership of the `direct` group was established by matching station identifiers against the
released `ushcn_daily` files. That establishes that the STATION is in the corpus. It does not
establish that our particular 2010--2014 values for that station are among the released rows: the
released slice could cover different years, or have gaps where we have data. The bound rests on
`direct` being data the model actually trained on, so the claim has to be checked against the
released values themselves.

WHAT THIS DOES. For each of the 300 `direct` windows, take its station and its exact date span,
pull the corresponding rows out of the released parquet, and compare values day by day. A window
is CONFIRMED only if the released file covers every one of its 576 days and the values agree.

Each unconfirmed window is classified by reason, because the reasons mean different things:
  no_station    the station is not in the released files at all
  short_cover   the released series does not span all 576 days (the released USHCN slice ends
                2014-09-28, so any window running past that date cannot be confirmed)
  missing_vals  the dates are covered but the released values are NaN where we have observations
  mismatch      the dates are covered and both have values, but they disagree

UNITS. The two sources are on different scales, and this has to be handled before anything is
called a mismatch. GHCN stores PRCP in tenths of a millimetre; the released files store hundredths
of an inch. The relation is exact -- released == round(ours / 2.54) held on every overlapping
observation checked -- so the comparison is made after that conversion rather than on raw values.
Skipping this step reports a 99.7% mismatch rate that is entirely an artifact of units.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .pool import load, FILES

TOL = 1e-6


def released_series():
    """{coop_id: (dates as YYYYMMDD int, PRCP values)} from the released parquet shards."""
    from huggingface_hub import hf_hub_download, list_repo_files
    import pyarrow.parquet as pq
    shards = sorted(f for f in list_repo_files("autogluon/chronos_datasets",
                                               repo_type="dataset")
                    if f.startswith("ushcn_daily/train-"))
    out = {}
    for sh in shards:
        p = hf_hub_download("autogluon/chronos_datasets", repo_type="dataset", filename=sh)
        t = pq.read_table(p, columns=["coop_id", "timestamp", "PRCP"])
        for row in t.to_pylist():
            d = pd.to_datetime(pd.Series(row["timestamp"]))
            out[int(row["coop_id"])] = (
                (d.dt.year * 10000 + d.dt.month * 100 + d.dt.day).to_numpy(np.int64),
                np.asarray(row["PRCP"], dtype=np.float64),
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="save the confirmed mask to files/direct_confirmed.npz")
    a = ap.parse_args()

    win, grp, sta, _ = load()
    z = np.load(FILES / "pool.npz", allow_pickle=True)
    span = z["span"]
    m = grp == "direct"
    W = win.shape[1]
    dw, ds, dsp = win[m], sta[m], span[m]
    print(f"[direct] {len(dw)} windows from {len(set(ds))} stations, {W} days each")

    rel = released_series()
    print(f"[released] {len(rel)} stations in ushcn_daily")
    allmax = max(int(v[0].max()) for v in rel.values())
    print(f"[released] latest date anywhere in the release: {allmax}")

    # GHCN tenths of a millimetre -> released hundredths of an inch. Verified rather than
    # assumed: the identity below is checked on the overlapping observations and reported.
    def to_released(v):
        return np.round(v.astype(np.float64) / 2.54)

    agree = []
    for w, s_, (aa, bb) in zip(dw, ds, dsp):
        cid = int(s_[3:])
        if cid not in rel:
            continue
        rd, rv = rel[cid]
        k = (rd >= aa) & (rd <= bb)
        if k.sum() < W // 2:
            continue
        theirs = rv[k]
        ok = np.isfinite(theirs)
        if ok.sum() > 20:
            agree.append(float((to_released(w[: k.sum()])[ok] == theirs[ok]).mean()))
        if len(agree) > 40:
            break
    print(f"[units] released == round(ours / 2.54) holds on "
          f"{np.mean(agree):.1%} of overlapping observations "
          f"(n={len(agree)} windows sampled)")

    reason = np.array(["confirmed"] * len(dw), dtype=object)
    for i, (w, s, (aa, bb)) in enumerate(zip(dw, ds, dsp)):
        cid = int(s[3:])
        if cid not in rel:
            reason[i] = "no_station"
            continue
        rd, rv = rel[cid]
        want = pd.date_range(str(aa), str(bb))
        want = (want.year * 10000 + want.month * 100 + want.day).to_numpy(np.int64)
        pos = np.searchsorted(rd, want)
        inb = (pos < len(rd)) & (rd[np.clip(pos, 0, len(rd) - 1)] == want)
        if not inb.all():
            reason[i] = "short_cover"
            continue
        theirs = rv[pos]
        if not np.isfinite(theirs).all():
            reason[i] = "missing_vals"
            continue
        if not np.array_equal(to_released(w), theirs):
            reason[i] = "mismatch"

    counts = {r: int((reason == r).sum()) for r in
              ["confirmed", "short_cover", "missing_vals", "mismatch", "no_station"]}
    n = len(dw)
    print("\n=== window-level verification of `direct` ===")
    for r, c in counts.items():
        print(f"  {r:14} {c:4}  ({c / n:5.1%})")
    print(f"\nmatch rate: {counts['confirmed']}/{n} = {counts['confirmed'] / n:.1%}")

    conf = reason == "confirmed"
    if conf.any():
        print(f"confirmed windows span {int(dsp[conf][:, 0].min())}--{int(dsp[conf][:, 1].max())} "
              f"over {len(set(ds[conf]))} stations")
    if (~conf).any():
        bad = dsp[~conf]
        print(f"unconfirmed windows span {int(bad[:, 0].min())}--{int(bad[:, 1].max())}")

    json.dump({"n_direct": n, "counts": counts,
               "match_rate": counts["confirmed"] / n,
               "unit_relation": "released == round(ours / 2.54)",
               "release_last_date": allmax},
              open(FILES / "direct_verification.json", "w"), indent=2)
    if a.write:
        np.savez(FILES / "direct_confirmed.npz", confirmed=conf, reason=reason.astype(str),
                 station=ds, span=dsp)
        print(f"\nwrote files/direct_confirmed.npz ({int(conf.sum())} confirmed)")
        write_keep_mask(grp, conf)


def write_keep_mask(grp, conf):
    """The mask every analysis applies: all windows, minus the direct ones not confirmed here."""
    keep = np.ones(len(grp), dtype=bool)
    keep[np.flatnonzero(grp == "direct")[~conf]] = False
    np.savez(FILES / "keep_mask.npz", keep=keep,
             note="direct windows verified value-by-value against the released ushcn_daily rows; "
                  "windows that could not be confirmed are excluded from every analysis")
    print(f"wrote files/keep_mask.npz ({int(keep.sum())} of {len(keep)} kept)")


if __name__ == "__main__":
    main()
