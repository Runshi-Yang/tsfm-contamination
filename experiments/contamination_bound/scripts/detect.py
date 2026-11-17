"""Loss-based membership detectors: on the benchmark, and on planted data where the answer is known.

    python -m experiments.contamination_bound.scripts.detect

WHY. The bound says how much membership could have helped; this asks the detection question
directly with the standard loss-based tools, so the paper can say what they see and whether they
would have seen anything. A detector that is silent on the benchmark means little unless it fires
on data we know was memorised, so every detector is also run on the planted checkpoint.

THE FOUR SCORES (higher = judged more likely a member):
  loss        -CE summed over the horizon. The plain loss attack.
  min-k       minus the mean of the 20% largest per-token CEs (Min-K%, Shi et al. 2024).
  diff-cal    minus the residual of CE after regressing it on the model-free difficulty d, so a
              window is judged by how much better it is fitted than its difficulty predicts.
  ref-cal     minus the gap between this model's standardised CE and a reference model's
              standardised loss on the same window. On the benchmark the reference is Moirai small,
              whose corpus has no USHCN gauges; on planted data it is the released checkpoint the
              planted one was trained from.

Benchmark: direct (member) vs indirect, AUROC with a station-bootstrap 95% interval, for the 46M and
709M released checkpoints. Planted: cores of each form and dose vs the 400 never-planted controls,
on the checkpoint saved by inject.py (the same weights the probe and the injection table use).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

from .pool import load, FILES
from .inject import OUT, KS, TYPES
from .bound import metric_rng

CTX, HOR, K = 512, 64, 0.2
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def auroc(pos, neg):
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def difficulty(win):
    ctx, hor = win[:, :CTX].astype(np.float64), win[:, CTX:CTX + HOR].astype(np.float64)
    den = np.abs(ctx[:, 1:] - ctx[:, :-1]).mean(1)
    return np.where(den > 1e-9, np.abs(hor).mean(1) / np.where(den > 1e-9, den, 1), np.nan)


@torch.no_grad()
def per_token_ce(model_id, win, batch=32):
    from chronos import ChronosPipeline
    pipe = ChronosPipeline.from_pretrained(model_id, device_map=DEV, torch_dtype=torch.float32)
    out = []
    for i in range(0, len(win), batch):
        c, h = torch.as_tensor(win[i:i + batch, :CTX]), torch.as_tensor(win[i:i + batch, CTX:CTX + HOR])
        ids, mask, scale = pipe.tokenizer.context_input_transform(c)
        lab, lmask = pipe.tokenizer.label_input_transform(h, scale)
        ids, mask, lab, lmask = ids.to(DEV), mask.to(DEV), lab.to(DEV), lmask.to(DEV)
        o = pipe.model.model(input_ids=ids, attention_mask=mask, labels=lab.masked_fill(~lmask, -100))
        ce = F.cross_entropy(o.logits.transpose(1, 2), lab, reduction="none") * lmask
        out.append(ce.float().cpu().numpy())
    del pipe; torch.cuda.empty_cache()
    return np.concatenate(out)


def scores(tok, d, ref):
    ce = tok.sum(1)
    k = max(1, int(round(K * tok.shape[1])))
    mink = -np.sort(tok, 1)[:, -k:].mean(1)
    ok = np.isfinite(d)
    X = np.column_stack([np.ones(ok.sum()), d[ok]])
    beta, *_ = np.linalg.lstsq(X, ce[ok], rcond=None)
    res = np.full(len(ce), np.nan); res[ok] = ce[ok] - X @ beta
    z = lambda v: (v - np.nanmean(v)) / np.nanstd(v)
    return {"loss": -ce, "min-k": mink, "diff-cal": -res, "ref-cal": -(z(ce) - z(ref))}


def main(n_boot=2000, seed=0):
    out = {}
    # ---------------------------------------------------------------- benchmark
    win, grp, sta, _ = load()
    keep = np.load(FILES / "keep_mask.npz")["keep"]
    win, grp, sta = win[keep].astype(np.float32), grp[keep], sta[keep]
    sel = np.isin(grp, ["direct", "indirect"])
    ref = np.load(FILES / "famscore_moirai-small.npz")["loss"][sel]
    w, g, s = win[sel], grp[sel], sta[sel]
    d = difficulty(w)
    uniq, inv = np.unique(s, return_inverse=True)
    by = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    for size in ("small", "large"):
        tok = per_token_ce(f"amazon/chronos-t5-{size}", w)
        sc = scores(tok, d, ref)
        rng = metric_rng(seed, "detect", size, "boot")
        res = {}
        for name, v in sc.items():
            ok = np.isfinite(v)
            a = auroc(v[(g == "direct") & ok], v[(g == "indirect") & ok])
            bs = []
            for _ in range(n_boot):
                idx = np.concatenate([by[p] for p in rng.integers(0, len(by), len(by))])
                vi, gi = v[idx], g[idx]
                m = np.isfinite(vi)
                bs.append(auroc(vi[(gi == "direct") & m], vi[(gi == "indirect") & m]))
            res[name] = dict(auroc=a, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
            print(f"[benchmark chronos-{size}] {name:9} AUROC {a:.3f} "
                  f"[{res[name]['ci'][0]:.3f}, {res[name]['ci'][1]:.3f}]", flush=True)
        out[f"benchmark/chronos-{size}"] = res

    # ---------------------------------------------------------------- planted (positive control)
    z = np.load(OUT / "inject_pool.npz", allow_pickle=True)
    probe = np.concatenate([z["core"], z["clean"]]).astype(np.float32)
    nc = len(z["core"])
    d = difficulty(probe)
    before = per_token_ce("amazon/chronos-t5-small", probe).sum(1)
    tok = per_token_ce(str(OUT / "contaminated_ep10"), probe)
    sc = scores(tok, d, before)
    res = {}
    for name, v in sc.items():
        for t in TYPES:
            for k in KS:
                m = (z["core_type"] == t) & (z["core_k"] == k)
                res[f"{name}/{t}/k{k}"] = auroc(v[:nc][m], v[nc:])
        print(f"[planted] {name:9} exact k=1..32: "
              + " ".join(f"{res[f'{name}/verbatim/k{k}']:.2f}" for k in KS)
              + " | blend k=32: " + f"{res[f'{name}/mixup/k32']:.2f}", flush=True)
    out["planted/chronos-small"] = res
    json.dump(out, open(FILES / "detect.json", "w"), indent=2)


if __name__ == "__main__":
    main()
