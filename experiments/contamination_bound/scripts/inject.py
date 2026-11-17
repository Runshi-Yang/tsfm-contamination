"""Positive control: inject known windows at a known repetition count and retrain.

    python -m experiments.contamination_bound.scripts.inject --build
    python -m experiments.contamination_bound.scripts.inject --train --epochs 10

WHY THIS EXISTS. The benchmark result is a null, and a null is only evidence if the same
instruments visibly fire when the effect IS present. This run manufactures the effect at a
known dose, so "we measured nothing on the released checkpoint" can be separated from "our
measurement cannot see anything".

It also supplies the exposure margin: the repetition count k at which membership first
becomes readable, against which a released checkpoint's roughly-one exposure is compared.

SAME DATA, SAME UNITS AS THE BENCHMARK. Background, core and clean windows are all GHCN
daily precipitation, 576 days (512 context + 64 horizon), drawn from stations DISJOINT
from the evaluation pool. The earlier version of this experiment used traffic and weather
data, which meant the positive control and the null were denominated in different units
and could not be quoted side by side. Here they can.

THE GRID, all inside ONE training run so that no cell is confounded by run-to-run variance:
each core window is assigned one (type, k) cell and appears in the corpus k times.

  k      1, 2, 4, 8, 16, 32 copies
  type   verbatim  identical copies -- the easiest possible contamination
         variant   rho = 0.99 near-duplicates, spectrum- and marginal-matched, so the eval
                   window itself never appears, only correlated versions of it
         mixup     convex blends with a random other series, which is what Chronos's own
                   TSMixup augmentation actually produces. The core never appears alone.

WHY THE THREE TYPES MATTER FOR THE HEADLINE NUMBER. Any single k quoted as "the exposure
at which memorisation appears" is a statement about ONE of these. `verbatim` is the easiest
case, so the k it gives is a LOWER bound on the exposure a realistically-augmented corpus
would need. Quoting it without the type attached would make the margin look tighter than
it is.

GROUND TRUTH IS A LOSS DIFFERENCE, NOT THE INJECTION LABEL. "was injected" is not the same
as "was memorised": a blended copy can appear 32 times and leave no trace. So the target is
delta(z) = loss_released(z) - loss_contaminated(z), measured per window against the
untouched released checkpoint. Only delta says whether exposure turned into memorisation.
"""
from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .pool import prcp_cache, inventory, windows, ushcn_corpus_ids

FILES = Path(__file__).resolve().parents[1] / "files"
OUT = Path(os.environ.get("CB_OUT", "outputs"))   # planted pool, per-window deltas, weights
CTX, HOR, W = 512, 64, 576
KS = (1, 2, 4, 8, 16, 32)
TYPES = ("verbatim", "variant", "mixup")


def aaft_surrogate(x, rng):
    """Phase-randomised surrogate: same amplitude spectrum, same marginal, no structure."""
    n = len(x)
    f = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2 * np.pi, len(f))
    ph[0] = 0
    if n % 2 == 0:
        ph[-1] = 0
    s = np.fft.irfft(np.abs(f) * np.exp(1j * ph), n)
    return np.sort(x)[np.argsort(np.argsort(s))]      # rank-match back to x's marginal


def make_copy(core, kind, rng, pool):
    if kind == "verbatim":
        return core.copy()
    if kind == "variant":
        z = (core - core.mean()) / (core.std() + 1e-8)
        s = aaft_surrogate(z, rng)
        v = 0.99 * z + np.sqrt(1 - 0.99 ** 2) * s
        return (v * core.std() + core.mean()).astype(np.float32)
    if kind == "mixup":
        lam = rng.uniform(0.3, 0.7)
        other = pool[rng.integers(len(pool))]
        return (lam * core + (1 - lam) * other).astype(np.float32)
    raise ValueError(kind)


def build(n_back=8000, n_core=180, n_clean=400, n_cand=20000, seed=0):
    """Background / core / clean windows from stations disjoint from the eval pool."""
    rng = np.random.default_rng(seed)
    inv = inventory()
    used = set()
    p = FILES / "pool.npz"
    if p.exists():
        used = set(np.load(p, allow_pickle=True)["station"].tolist())
    cand = sorted(s for s, (la, lo, y0, y1) in inv.items()
                  if s.startswith("US") and s not in used and y0 <= 2010 and y1 >= 2014)
    rng.shuffle(cand)
    cand = set(cand[:n_cand])
    print(f"[inject] {len(cand)} candidate stations, disjoint from the eval pool")

    import pandas as pd
    df = pd.concat([prcp_cache(y) for y in (2010, 2011, 2012, 2013, 2014)])
    df = df[df.id.isin(cand)].sort_values(["id", "date"])
    ser = {s: (g.date.to_numpy(), g.value.to_numpy(np.float32))
           for s, g in df.groupby("id", sort=False)}
    w, st, sp = windows(ser, W, 3, rng)
    print(f"[inject] {len(w)} windows from {len(set(st))} stations")

    # Cores and the held-out clean set are fixed; the background takes whatever is left. Strict
    # contiguity limits the supply of windows, and failing outright on a shortfall would make the
    # experiment hostage to that yield. A smaller background raises the share of the corpus that
    # is injected, which makes memorisation EASIER to produce, so any exposure threshold measured
    # here is a lower bound on what a realistically diluted corpus would need. The share is
    # printed so it can be quoted alongside the threshold.
    idx = rng.permutation(len(w))
    assert len(w) >= n_core + n_clean + 500, f"only {len(w)} windows; too few to run"
    n_back = min(n_back, len(w) - n_core - n_clean)
    need = n_back + n_core + n_clean
    back, core, clean = np.split(idx[:need], [n_back, n_back + n_core])
    OUT.mkdir(parents=True, exist_ok=True)

    cells = [(t, k) for t in TYPES for k in KS]
    assign = np.array([cells[i % len(cells)] for i in range(n_core)], dtype=object)
    np.savez(OUT / "inject_pool.npz", back=w[back], core=w[core], clean=w[clean],
             back_st=st[back], core_st=st[core], clean_st=st[clean],
             core_type=np.array([a[0] for a in assign]),
             core_k=np.array([int(a[1]) for a in assign]))
    n_copies = sum(int(a[1]) for a in assign)
    print(f"[inject] background {len(back)} | cores {len(core)} over {len(cells)} cells "
          f"| clean {len(clean)}")
    print(f"[inject] {n_copies} injected copies -> corpus {len(back) + n_copies}, of which "
          f"{n_copies / (len(back) + n_copies):.1%} is injected "
          f"(real pretraining is far more dilute; this setting favours memorisation)")
    for t in TYPES:
        print(f"  {t:9} " + " ".join(
            f"k={k}:{int(((np.array([a[0] for a in assign]) == t) & (np.array([int(a[1]) for a in assign]) == k)).sum())}"
            for k in KS))


def corpus_from_pool(seed=0):
    z = np.load(OUT / "inject_pool.npz", allow_pickle=True)
    rng = np.random.default_rng(seed)
    rows = [z["back"]]
    for core, t, k in zip(z["core"], z["core_type"], z["core_k"]):
        rows.append(np.stack([make_copy(core, str(t), rng, z["back"]) for _ in range(int(k))]))
    corpus = np.concatenate(rows).astype(np.float32)
    rng.shuffle(corpus)
    print(f"[corpus] {len(corpus)} windows "
          f"({len(z['back'])} background + {len(corpus) - len(z['back'])} injected copies)")
    return corpus, z


@torch.no_grad()
def per_window_ce(pipe, win, device, batch=32):
    out = []
    for i in range(0, len(win), batch):
        c = torch.as_tensor(win[i:i + batch, :CTX])
        h = torch.as_tensor(win[i:i + batch, CTX:CTX + HOR])
        ids, mask, scale = pipe.tokenizer.context_input_transform(c)
        lab, lmask = pipe.tokenizer.label_input_transform(h, scale)
        o = pipe.model.model(input_ids=ids.to(device), attention_mask=mask.to(device),
                             labels=lab.to(device).masked_fill(~lmask.to(device), -100))
        ce = F.cross_entropy(o.logits.transpose(1, 2), lab.to(device), reduction="none")
        out.append((ce * lmask.to(device)).sum(-1).float().cpu().numpy())
    return np.concatenate(out)


def train(epochs=10, lr=1e-4, batch=16, model="amazon/chronos-t5-small", seed=0):
    from chronos import ChronosPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus, z = corpus_from_pool(seed)

    base = ChronosPipeline.from_pretrained(model, device_map=device,
                                           torch_dtype=torch.float32)
    probe = np.concatenate([z["core"], z["clean"]])
    ce_before = per_window_ce(base, probe, device)
    del base
    torch.cuda.empty_cache()

    pipe = ChronosPipeline.from_pretrained(model, device_map=device,
                                           torch_dtype=torch.float32)
    net = pipe.model.model
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    net.train()
    for ep in range(epochs):
        order = rng.permutation(len(corpus))
        tot = n = 0.0
        for i in range(0, len(order), batch):
            b = corpus[order[i:i + batch]]
            c = torch.as_tensor(b[:, :CTX])
            h = torch.as_tensor(b[:, CTX:CTX + HOR])
            ids, mask, scale = pipe.tokenizer.context_input_transform(c)
            lab, lmask = pipe.tokenizer.label_input_transform(h, scale)
            o = net(input_ids=ids.to(device), attention_mask=mask.to(device),
                    labels=lab.to(device).masked_fill(~lmask.to(device), -100))
            opt.zero_grad(); o.loss.backward(); opt.step()
            tot += float(o.loss); n += 1
        print(f"[train] epoch {ep + 1}/{epochs} loss {tot / max(n, 1):.4f}", flush=True)

    net.eval()
    # Keep the contaminated weights. Step 5 needs a checkpoint whose memorisation is known and
    # measured, otherwise a probe that reports nothing cannot be told apart from a probe that
    # does not work.
    ckpt = OUT / f"contaminated_ep{epochs}"
    pipe.model.model.save_pretrained(ckpt)
    print(f"[train] saved contaminated checkpoint to {ckpt}", flush=True)
    ce_after = per_window_ce(pipe, probe, device)
    nc = len(z["core"])
    delta = ce_before - ce_after                     # positive = contaminated model fits better
    np.savez(OUT / f"inject_delta_ep{epochs}.npz", ce_before=ce_before, ce_after=ce_after,
             delta=delta, n_core=nc, core_type=z["core_type"], core_k=z["core_k"])

    d_core, d_clean = delta[:nc], delta[nc:]
    print(f"\n=== delta = CE(released) - CE(contaminated); clean baseline "
          f"{d_clean.mean():.2f} +- {d_clean.std():.2f} ===")
    hdr = f"{'type':10}" + "".join(f"{'k=' + str(k):>12}" for k in KS)
    print(hdr); print("-" * len(hdr))
    res = {}
    for t in TYPES:
        row = []
        for k in KS:
            m = (z["core_type"] == t) & (z["core_k"] == k)
            excess = float(d_core[m].mean() - d_clean.mean()) if m.any() else float("nan")
            res[f"{t}/k{k}"] = excess
            row.append(f"{excess:>12.2f}")
        print(f"{t:10}" + "".join(row))
    print("\n(excess delta over the clean baseline, in summed token CE. A cell is only "
          "'memorised' when this is clearly above the clean spread.)")
    json.dump(res, open(FILES / f"inject_ep{epochs}.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default="amazon/chronos-t5-small")
    ap.add_argument("--n-cand", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.build:
        build(seed=a.seed, n_cand=a.n_cand)
    if a.train:
        train(epochs=a.epochs, lr=a.lr, batch=a.batch, model=a.model, seed=a.seed)
