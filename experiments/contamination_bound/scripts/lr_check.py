"""Pick each family's fine-tuning learning rate before any planted data is looked at.

    python -m experiments.contamination_bound.scripts.lr_check --family moirai --size small

WHY. One learning rate for every family sounds fairer than it is. 1e-4 is a normal rate for
Chronos, but full fine-tuning of Moirai at 1e-4 makes its loss on unseen windows worse epoch after
epoch, with or without warm-up. A model whose training is breaking down will fail to memorise for a
reason that has nothing to do with exposure, and the dose-response would then measure the
learning rate. So each family gets the largest rate its fine-tuning tolerates.

THE RULE, fixed before the planted runs: from the grid {1e-4, 3e-5, 1e-5}, take the largest rate at
which two epochs of fine-tuning on 3000 background windows leaves the loss on 200 *other* background
windows no more than 1% above where it started. Only background windows are used -- no planted
copy, no core and none of the held-out control windows -- so the choice cannot be tuned, even
indirectly, towards or away from memorisation.

AMENDMENT (2026-09-29, applied to every family alike). A two-epoch check can miss instability that
appears later. TimesFM at 3e-5 passed it, then in the full ten-epoch run its epoch-mean training loss
jumped from about 1 to 294 and 1116 (200M) and to 644 and 2208 (500M); every other family's curve
fell smoothly. So a full run also has to be stable: no epoch's mean training loss may exceed three
times the first epoch's. A run that fails this is discarded and the family moves to the next rate in
the grid. Only TimesFM failed it, and was rerun at 1e-5.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .inject import OUT
from . import adapters

FILES = Path(__file__).resolve().parents[1] / "files"
GRID = (1e-4, 3e-5, 1e-5)
TOL = 0.01
WARMUP = 100


def trial(family, size, lr, tr, ev, epochs=2, batch=16, seed=0):
    torch.manual_seed(seed)
    ad = adapters.REGISTRY[family](size)
    opt = torch.optim.AdamW(ad.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP))
    ad.eval(); start = float(ad.window_loss(ev).mean())
    rng, curve = np.random.default_rng(seed), []
    for _ in range(epochs):
        ad.train()
        o = rng.permutation(len(tr))
        for i in range(0, len(o), batch):
            loss = ad.batch_loss(tr[o[i:i + batch]])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ad.parameters(), 1.0)
            opt.step(); sch.step()
        ad.eval(); curve.append(float(ad.window_loss(ev).mean()))
    del ad; torch.cuda.empty_cache()
    return start, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--size", required=True)
    a = ap.parse_args()
    back = np.load(OUT / "inject_pool.npz")["back"].astype(np.float32)
    tr, ev = back[:3000], back[3000:3200]
    res, chosen = {}, None
    for lr in GRID:
        start, curve = trial(a.family, a.size, lr, tr, ev)
        rel = curve[-1] / start - 1 if start > 0 else (curve[-1] - start) / abs(start)
        ok = rel <= TOL
        res[str(lr)] = dict(start=start, curve=curve, rel_change=rel, ok=ok)
        print(f"lr={lr:g}: held-out background loss {start:.4f} -> "
              + " -> ".join(f"{c:.4f}" for c in curve) + f"  ({rel:+.1%})  {'ok' if ok else 'too high'}",
              flush=True)
        if ok:
            chosen = lr
            break
    print(f"chosen lr for {a.family}-{a.size}: {chosen}")
    json.dump(dict(rule=f"largest of {GRID} with <= {TOL:.0%} rise in held-out background loss "
                        f"after 2 epochs", chosen=chosen, trials=res),
              open(FILES / f"lr_{a.family}-{a.size}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
