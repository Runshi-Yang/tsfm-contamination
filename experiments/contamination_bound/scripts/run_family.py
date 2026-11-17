"""Angle 2 across model families: plant data at a known dose, retrain, see what gets memorised.

    python -m experiments.contamination_bound.scripts.run_family --family chronos --size large

WHY THIS EXISTS. `inject.py` runs the dose-response on one Chronos checkpoint. The question it
answers -- how many exposures a window needs before testing on it is unfair -- is only interesting
if it is not a property of one architecture. This runs the identical experiment on other families.

WHAT IS HELD FIXED ACROSS FAMILIES. The corpus (the same `inject_pool.npz`: 8000 background
windows, 180 cores planted at k in {1,2,4,8,16,32} as verbatim / variant / mixup copies, 400 held
out), the schedule (10 epochs, batch 16, AdamW, 100-step linear warm-up, gradient clipping at 1), the window shape (512 context + 64
horizon) and the decision rule. The learning rate is the one exception: each family uses the
largest rate its fine-tuning tolerates, chosen on background windows only by lr_check.py.

WHAT IS NOT. Each family is scored with its own training objective on the 64-step horizon: token
cross-entropy for Chronos, negative log-likelihood of the predictive distribution for Moirai and
Lag-Llama, quantile loss for TimesFM (which has no likelihood). That is deliberate. The decision
rule compares each planted cell against never-planted windows scored by the same model before and
after the same fine-tune, so the unit of the loss cancels; what we need is that the loss is the
thing the model was actually optimising, because that is where memorisation would show.

DECISION RULE, as in `exposure.py` and the paper: delta = loss_before - loss_after per window;
a cell is readable when a two-sided Welch test of its cores against the held-out windows gives
p < alpha and the excess is positive. Both alpha = 0.05 and the Bonferroni alpha = 0.05/18 are
recorded; the paper quotes Bonferroni.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from .inject import OUT, KS, TYPES, corpus_from_pool
from . import adapters

FILES = Path(__file__).resolve().parents[1] / "files"


def analyse(delta, nc, ctype, ck, epochs):
    d_core, d_clean = delta[:nc], delta[nc:]
    base, p95 = float(d_clean.mean()), float(np.percentile(d_clean, 95))
    bonf = 0.05 / (len(TYPES) * len(KS))
    cells, first = {}, {}
    for t in TYPES:
        for k in KS:
            m = (ctype == t) & (ck == k)
            v = d_core[m]
            excess = float(v.mean() - base)
            p = float(stats.ttest_ind(v, d_clean, equal_var=False).pvalue)
            cells[f"{t}/k{k}"] = dict(
                exposures=int(k) * epochs, excess=excess, median=float(np.median(v) - base),
                p=p, frac=float((v > p95).mean()),
                readable=bool(p < 0.05 and excess > 0),
                readable_bonf=bool(p < bonf and excess > 0))
            if cells[f"{t}/k{k}"]["readable_bonf"] and t not in first:
                first[t] = int(k) * epochs
    return dict(clean_mean=base, clean_sd=float(d_clean.std(ddof=1)), clean_p95=p95,
                bonferroni_alpha=bonf, first_readable_exposures_bonf=first, cells=cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(adapters.REGISTRY))
    ap.add_argument("--size", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None,
                    help="default: the rate chosen by lr_check.py for this family and size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the adapter")
    ap.add_argument("--per-epoch", action="store_true",
                    help="score the probe windows after every epoch, so a core planted k times is\n"
                         "measured at k, 2k, ..., 10k exposures within ONE run (reaches 1-10 exposures)")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    tag = f"{a.family}-{a.size}"
    if a.lr is not None:                      # an explicit rate is a sensitivity run: keep it apart
        tag += f"-lr{a.lr:g}"
    lr_tag = tag.split("-lr")[0]
    if a.lr is None:
        a.lr = json.load(open(FILES / f"lr_{lr_tag}.json"))["chosen"]
        assert a.lr is not None, f"lr_check found no stable rate for {tag}"
    out = OUT / "family" / tag
    out.mkdir(parents=True, exist_ok=True)
    corpus, z = corpus_from_pool(a.seed)
    probe = np.concatenate([z["core"], z["clean"]]).astype(np.float32)
    if a.smoke:
        corpus, probe = corpus[:64], probe[:64]

    ad = adapters.REGISTRY[a.family](a.size)
    t0 = time.time()
    ad.eval()
    before = ad.window_loss(probe)
    print(f"[{tag}] loss before: mean {before.mean():.4f}  ({time.time() - t0:.0f}s)", flush=True)

    opt = torch.optim.AdamW(ad.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100))
    rng = np.random.default_rng(a.seed)
    per_epoch = []
    for ep in range(1 if a.smoke else a.epochs):
        ad.train()
        order = rng.permutation(len(corpus))
        tot = n = 0.0
        for i in range(0, len(order), a.batch):
            loss = ad.batch_loss(corpus[order[i:i + a.batch]])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ad.parameters(), 1.0)
            opt.step(); sch.step()
            tot += float(loss.detach()); n += 1
        print(f"[{tag}] epoch {ep + 1}/{a.epochs} train loss {tot / n:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if a.per_epoch and not a.smoke:
            ad.eval()
            per_epoch.append(before - ad.window_loss(probe))

    ad.eval()
    after = ad.window_loss(probe)
    delta = before - after
    print(f"[{tag}] loss after: mean {after.mean():.4f}", flush=True)
    if a.smoke:
        print("[smoke] adapter OK"); return

    nc = len(z["core"])
    np.savez(out / "delta.npz", before=before, after=after, delta=delta, n_core=nc,
             core_type=z["core_type"], core_k=z["core_k"])
    res = analyse(delta, nc, z["core_type"], z["core_k"], a.epochs)
    res.update(family=a.family, size=a.size, loss=ad.LOSS, epochs=a.epochs, lr=a.lr,
               batch=a.batch, n_params=ad.n_params(), seconds=round(time.time() - t0))
    if a.per_epoch:
        # exposures = k * epoch; one analysis per epoch, same rule, same controls
        res["per_epoch"] = {str(e + 1): analyse(dl, nc, z["core_type"], z["core_k"], e + 1)
                            for e, dl in enumerate(per_epoch)}
        np.savez(out / "delta_per_epoch.npz", delta=np.stack(per_epoch))
        tag += "-perepoch"
    json.dump(res, open(FILES / f"family_{tag}.json", "w"), indent=2)
    for t in TYPES:
        row = " ".join(f"{res['cells'][f'{t}/k{k}']['frac']:.1f}"
                       + ("*" if res['cells'][f'{t}/k{k}']['readable_bonf'] else " ")
                       for k in KS)
        print(f"  {t:9} frac by k: {row}")
    print(f"  first readable (Bonferroni), in exposures: {res['first_readable_exposures_bonf']}")


if __name__ == "__main__":
    main()
