"""Build the three-group GHCN precipitation pool from raw by-year files.

    python -m experiments.contamination_bound.scripts.pool --build

WHAT THE THREE GROUPS ARE, and why membership is decided by the calendar rather than by
trusting a corpus table.

  direct    USHCN stations that ARE in the Chronos pretraining corpus, 2010-2014.
            `ushcn_daily` is pretraining-only data and its loader takes PRCP, so these
            station-years are literally in the training set.
  indirect  US co-op stations NOT in USHCN, THE SAME YEARS. Different gauges, same
            weather: same fronts, same droughts, same ENSO phase.
  clean     THE SAME non-USHCN stations as `indirect`, from 2024-06-01 onward.
            chronos-t5-small was created 2024-02-21, so these observations did not exist
            when it was trained. Unseen because they had not happened yet.

`indirect vs clean` therefore varies ONE thing -- the period -- on a fixed set of gauges,
and `direct vs indirect` varies membership at a fixed period.

EVERYTHING IS THE SAME VARIABLE FROM THE SAME FILES. All three groups are GHCN daily PRCP
parsed out of the same by-year archives with identical preprocessing, so no unit, QC or
loader artifact can separate them. That is the property the earlier benchmarks in this
project lacked, and it is checked at build time by the model-free gate in `gate()`: with
the model left out entirely, no hand-crafted feature may separate a contrast above 0.60.
A contrast that fails the gate is disqualified -- the contrast, not just the method.

WINDOW LENGTH is 576 = 512 context + 64 horizon, which is exactly the chronos-t5
`context_length` / `prediction_length`, so a window is scored the way the model was trained.

GRADED TARGET. "contemporaneous" is an assumption about correlation, not a measurement, so
`corpus_corr` records how strongly each window actually correlates with contemporaneous
data from the nearest corpus stations. It is what makes the eventual bound quotable: the
direct-vs-indirect contrast bounds the increment between these two correlation levels, not
"leakage" in the abstract.

Quality control: GHCN rows carrying a non-empty Q-FLAG failed NOAA's own checks and are
dropped here. (The earlier version of this benchmark ignored the flag.)
"""
from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(os.environ.get("GHCN_CACHE", "ghcn_cache"))   # raw GHCN downloads
FILES = Path(__file__).resolve().parents[1] / "files"
INVENTORY = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt"
BYYEAR = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/{y}.csv.gz"

CORPUS_YEARS = [2010, 2011, 2012, 2013, 2014]   # ushcn_daily ends 2014-09-28
POST_YEARS = [2024, 2025, 2026]
POST_START = 20240601                            # margin after the 2024-02-21 release
COLS = ["id", "date", "elem", "value", "mflag", "qflag", "sflag", "obstime"]


# ----------------------------------------------------------------- corpus membership

def ushcn_corpus_ids():
    """GHCN ids of the USHCN stations actually present in the Chronos corpus."""
    from huggingface_hub import hf_hub_download, list_repo_files
    import pyarrow.parquet as pq
    shards = sorted(f for f in list_repo_files("autogluon/chronos_datasets",
                                               repo_type="dataset")
                    if f.startswith("ushcn_daily/train-"))
    ids = set()
    for sh in shards:
        t = pq.read_table(hf_hub_download("autogluon/chronos_datasets",
                                          repo_type="dataset", filename=sh),
                          columns=["coop_id"])
        ids |= {f"USC{int(c):08d}" for c in t.column("coop_id").to_pylist()}
    return ids


def inventory():
    """{station: (lat, lon, first_year, last_year)} for stations reporting PRCP."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / "ghcnd-inventory.txt"
    if not p.exists():
        urllib.request.urlretrieve(INVENTORY, p)
    rows = {}
    for line in open(p):
        if line[31:35].strip() != "PRCP":
            continue
        rows[line[:11]] = (float(line[12:20]), float(line[21:30]),
                           int(line[36:40]), int(line[41:45]))
    return rows


# ------------------------------------------------------------------------ raw series

def prcp_cache(year):
    """QC-passed PRCP rows for US stations, extracted once per year and cached.

    Decompressing and parsing a by-year file takes minutes, and every rebuild needs the
    same rows, so the expensive pass is done once and the result parked as parquet. The
    cache is keyed only by year -- it holds ALL US stations, so changing the station
    selection does not invalidate it."""
    q = CACHE / f"prcp_us_{year}.parquet"
    if q.exists():
        return pd.read_parquet(q)
    p = CACHE / f"{year}.csv.gz"
    if not p.exists():
        print(f"[ghcn] downloading {year}", flush=True)
        urllib.request.urlretrieve(BYYEAR.format(y=year), p)
    out = []
    for ch in pd.read_csv(p, header=None, names=COLS, usecols=[0, 1, 2, 3, 5],
                          dtype={"id": str, "date": np.int64, "elem": str,
                                 "value": "float32", "qflag": str},
                          chunksize=4_000_000, compression="gzip"):
        ch = ch[(ch.elem == "PRCP") & ch.qflag.isna() & ch.id.str.startswith("US")]
        if len(ch):
            out.append(ch[["id", "date", "value"]])
    df = (pd.concat(out) if out else
          pd.DataFrame({"id": pd.Series(dtype=str), "date": pd.Series(dtype=np.int64),
                        "value": pd.Series(dtype=np.float32)}))
    df.to_parquet(q, index=False)
    print(f"[ghcn] {year}: cached {len(df)} US PRCP rows", flush=True)
    return df


def read_year(year, keep):
    """PRCP rows for `keep` stations from one year."""
    df = prcp_cache(year)
    df = df[df.id.isin(keep)]
    print(f"[ghcn] {year}: {len(df):>9} clean PRCP rows / {df.id.nunique()} stations",
          flush=True)
    return df


def series(years, keep):
    """{station: (dates, values)} sorted by date."""
    df = pd.concat([read_year(y, keep) for y in years]).sort_values(["id", "date"])
    return {s: (g.date.to_numpy(), g.value.to_numpy(np.float32))
            for s, g in df.groupby("id", sort=False)}


def windows(ser, W, n_per, rng, date_min=None, min_std=1e-6):
    """Sample up to n_per non-overlapping contiguous windows per station.

    Contiguity is required: a window is kept only if its date span equals its length in
    days, so gaps never get silently concatenated into a fake series."""
    ws, sids, spans = [], [], []
    for sid, (d, v) in ser.items():
        if date_min is not None:
            m = d >= date_min
            d, v = d[m], v[m]
        if len(v) < W:
            continue
        starts = list(range(0, len(v) - W + 1))
        rng.shuffle(starts)
        used, got = [], 0
        for st in starts:
            if got >= n_per:
                break
            if any(abs(st - u) < W for u in used):        # keep windows disjoint
                continue
            w = v[st:st + W]
            if not np.isfinite(w).all() or w.std() <= min_std:
                continue
            a, b = int(d[st]), int(d[st + W - 1])
            if (pd.Timestamp(str(b)) - pd.Timestamp(str(a))).days != W - 1:
                continue                                  # gap inside the span
            ws.append(w); sids.append(sid); spans.append((a, b)); used.append(st); got += 1
    return (np.asarray(ws, np.float32), np.array(sids),
            np.asarray(spans, np.int64).reshape(-1, 2))


# ---------------------------------------------------------------------- graded target

def corpus_correlation(win, station, span, corpus_ser, latlon, k=5):
    """Max |corr| against contemporaneous data from the k nearest corpus stations.

    Returns NaN, not 0, when no corpus station has data overlapping the window's dates.
    Those are different statements: 0 would mean "measured, and uncorrelated", while NaN
    means "the corpus holds nothing from this period, so the quantity is undefined". The
    post-release group is entirely in the second case -- the corpus ends in 2014 -- and
    reporting it as a measured 0.000 overstates what was computed."""
    ids = [s for s in corpus_ser]
    P = np.array([latlon[s] for s in ids])
    out = np.full(len(win), np.nan, np.float32)
    for i, (sid, (a, b)) in enumerate(zip(station, span)):
        if sid not in latlon:
            continue
        la, lo = latlon[sid]
        near = np.argsort((P[:, 0] - la) ** 2 + (P[:, 1] - lo) ** 2)[:k]
        best, compared = 0.0, False
        for j in near:
            dt, vt = corpus_ser[ids[j]]
            m = (dt >= a) & (dt <= b)
            if m.sum() < len(win[i]) // 2:
                continue
            n = min(int(m.sum()), len(win[i]))
            c = np.corrcoef(vt[m][:n], win[i][:n])[0, 1]
            if np.isfinite(c):
                best = max(best, abs(float(c)))
                compared = True
        out[i] = best if compared else np.nan
    return out


# ------------------------------------------------------------------- model-free gate

def features(w):
    """Hand-crafted descriptors, computed WITHOUT the model, on z-normalised windows.

    Windows are z-normalised per window first: raw scale encodes dataset/units convention
    and separating on it is an artifact, not a signal (this bug inflated an earlier
    version of this benchmark)."""
    z = (w - w.mean(1, keepdims=True)) / (w.std(1, keepdims=True) + 1e-8)
    d = np.diff(z, axis=1)
    S = np.abs(np.fft.rfft(z, axis=1)) ** 2
    S = S[:, 1:] / (S[:, 1:].sum(1, keepdims=True) + 1e-12)
    nb = S.shape[1]
    edges = [0, .02, .05, .10, .25, .50, 1.0]
    bands = [S[:, int(a * nb):max(int(b * nb), int(a * nb) + 1)].sum(1)
             for a, b in zip(edges[:-1], edges[1:])]
    acf1 = (z[:, :-1] * z[:, 1:]).mean(1)
    ent = -(S * np.log(S + 1e-12)).sum(1) / np.log(nb)
    frac0 = (w <= 0).mean(1)
    return (np.column_stack(bands + [np.abs(d).mean(1), d.std(1), acf1, ent, frac0]),
            [f"band{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])]
            + ["mean|diff|", "std(diff)", "acf1", "spec_entropy", "frac_dry"])


def auroc(score, y):
    from scipy.stats import rankdata
    r = rankdata(score)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def gate(w, g, thresh=0.60):
    F, names = features(w)
    print(f"\nMODEL-FREE GATE (no model loaded). A usable contrast must stay under {thresh}.")
    ok = True
    for pos, neg in [("indirect", "clean"), ("direct", "clean"), ("direct", "indirect")]:
        m = np.isin(g, [pos, neg]); y = g[m] == pos
        a = [max(auroc(F[m][:, i], y), 1 - auroc(F[m][:, i], y)) for i in range(F.shape[1])]
        j = int(np.argmax(a))
        flag = "PASS" if max(a) < thresh else "FAIL"
        ok &= max(a) < thresh
        print(f"  {pos:9} vs {neg:9}  worst feature {names[j]:14} {max(a):.3f}   {flag}")
    return ok


# ------------------------------------------------------------------------------ build

def build(W=576, n_cand=1500, n_direct=300, n_indirect=300, n_clean=2400, seed=0):
    rng = np.random.default_rng(seed)
    inv = inventory()
    ushcn = ushcn_corpus_ids()
    print(f"[corpus] {len(ushcn)} USHCN stations in the Chronos corpus")

    corpus_st = {s for s in ushcn if s in inv and inv[s][2] <= 2010 and inv[s][3] >= 2014}
    cand = sorted(s for s, (la, lo, y0, y1) in inv.items()
                  if s.startswith("USC00") and s not in ushcn and y0 <= 2010 and y1 >= 2025)
    rng.shuffle(cand)
    other_st = set(cand[:n_cand])
    print(f"[stations] direct pool {len(corpus_st)} | indirect/clean pool {len(other_st)}")

    old = series(CORPUS_YEARS, corpus_st | other_st)
    new = series(POST_YEARS, other_st)
    Dw, Ds, Dsp = windows({s: v for s, v in old.items() if s in corpus_st}, W, 2, rng)
    Iw, Is, Isp = windows({s: v for s, v in old.items() if s in other_st}, W, 2, rng)
    Cw, Cs, Csp = windows(new, W, 3, rng, date_min=POST_START)
    print(f"[raw] direct {len(Dw)} | indirect {len(Iw)} | clean {len(Cw)}")

    # `indirect` and `clean` must be the SAME stations -- that is what makes the decisive
    # contrast vary only the period. Restrict both to the intersection before subsampling.
    shared = set(Is) & set(Cs)
    print(f"[shared] {len(shared)} stations appear in both indirect and clean")
    ki, kc = np.isin(Is, list(shared)), np.isin(Cs, list(shared))
    Iw, Is, Isp = Iw[ki], Is[ki], Isp[ki]
    Cw, Cs, Csp = Cw[kc], Cs[kc], Csp[kc]

    def take(w, s, sp, n):
        if len(w) <= n:
            return w, s, sp
        k = rng.choice(len(w), n, replace=False)
        return w[k], s[k], sp[k]
    Dw, Ds, Dsp = take(Dw, Ds, Dsp, n_direct)
    Iw, Is, Isp = take(Iw, Is, Isp, n_indirect)
    Cw, Cs, Csp = take(Cw, Cs, Csp, n_clean)

    win = np.concatenate([Dw, Iw, Cw])
    grp = np.array(["direct"] * len(Dw) + ["indirect"] * len(Iw) + ["clean"] * len(Cw))
    sta = np.concatenate([Ds, Is, Cs])
    spn = np.concatenate([Dsp, Isp, Csp])

    latlon = {s: (inv[s][0], inv[s][1]) for s in inv}
    corr = corpus_correlation(win, sta, spn, {s: old[s] for s in corpus_st if s in old},
                              latlon)

    FILES.mkdir(parents=True, exist_ok=True)
    np.savez(FILES / "pool.npz", windows=win, group=grp, station=sta, span=spn,
             corpus_corr=corr)
    print(f"\n[pool] {len(win)} windows, W={W}")
    for g in ["direct", "indirect", "clean"]:
        m = grp == g
        print(f"  {g:9} n={int(m.sum()):5} stations={len(set(sta[m])):5} "
              f"corpus_corr p50={np.median(corr[m]):.3f}  "
              f"dry-day frac={float((win[m] <= 0).mean()):.3f}")
    passed = gate(win, grp)
    print(f"\nGATE {'PASSED' if passed else 'FAILED'}")
    return win, grp, sta, corr


def load():
    z = np.load(FILES / "pool.npz", allow_pickle=True)
    return z["windows"], z["group"], z["station"], z["corpus_corr"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--window", type=int, default=576)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-cand", type=int, default=10_000)
    ap.add_argument("--n-direct", type=int, default=300)
    ap.add_argument("--n-indirect", type=int, default=300)
    ap.add_argument("--n-clean", type=int, default=3000)
    a = ap.parse_args()
    build(W=a.window, seed=a.seed, n_cand=a.n_cand, n_direct=a.n_direct,
          n_indirect=a.n_indirect, n_clean=a.n_clean)
