"""Does Moirai's null come from its loss? Rescore the planted Moirai run with point losses.

    python -m experiments.contamination_bound.scripts.moirai_point

WHY THIS EXISTS. Moirai small is the one family in which no planted cell is detectable, even exact
copies at 320 exposures. Two readings fit that: Moirai memorises less, or its mixture likelihood
does not expose the trace to our tests. A forecaster that has memorised a window should also put its
forecasts closer to that window's horizon, so scoring the same windows with point losses separates
the two. The planted Moirai weights were not kept, so this retrains them.

WHAT IS HELD FIXED. Everything in run_family.py: the corpus, seed, data order, learning rate (from
lr_check.py), schedule and decision rule (run_family.analyse). The NLL is scored again before and
after, and its deltas are compared with the saved run's, to check that the retrained model is the
same one the paper reports. The point losses are scored inside a forked RNG, so the extra sampling
does not change the training run.

THE POINT LOSSES, fixed before looking at the result, all on the 64-day horizon and divided by the
context's standard deviation + 1, as in the TimesFM adapter:
  mae     absolute error of the median forecast (closest to MASE)
  pinball pinball loss averaged over the nine deciles
  sqpin   squared error of the mean forecast + pinball, TimesFM's own training loss
Forecasts are 500 sample paths; before and after use the same random draws per batch.
Point losses barely respond to the model on this data (Section 2.4), so a null here is expected and
does not by itself mean Moirai memorises less.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from .inject import OUT, KS, TYPES, corpus_from_pool
from .run_family import FILES, analyse
from . import adapters

SAMPLES = 500
POINT = ("mae", "pinball", "sqpin")


@torch.no_grad()
def point_losses(ad, win, batch=32, seed=0):
    """Per-window point losses from SAMPLES forecast paths. Each batch reseeds inside a forked RNG,
    so the same windows get the same draws before and after training."""
    q = torch.linspace(0.1, 0.9, 9, device=adapters.DEV)
    out = {k: [] for k in POINT}
    for i in range(0, len(win), batch):
        x = torch.as_tensor(win[i:i + batch], device=adapters.DEV)
        ctx, y = x[:, :adapters.CTX], x[:, adapters.CTX:]
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed + i)
            paths = ad.fc(past_target=ctx.unsqueeze(-1),
                          past_observed_target=torch.ones_like(ctx, dtype=torch.bool).unsqueeze(-1),
                          past_is_pad=torch.zeros_like(ctx, dtype=torch.bool),
                          num_samples=SAMPLES)                          # [B, S, 64]
        s = ctx.std(1, keepdim=True) + 1.0
        yn, pn = y / s, paths / s.unsqueeze(1)
        qs = torch.quantile(pn, q, dim=1).permute(1, 2, 0)              # [B, 64, 9]
        err = yn.unsqueeze(-1) - qs
        pin = torch.maximum(q * err, (q - 1) * err).mean((-1, -2))
        out["mae"].append((qs[..., 4] - yn).abs().mean(-1))
        out["pinball"].append(pin)
        out["sqpin"].append(((pn.mean(1) - yn) ** 2).mean(-1) + pin)
    return {k: torch.cat(v).cpu().numpy() for k, v in out.items()}


def main():
    seed, epochs, bs = 0, 10, 16
    torch.manual_seed(seed)
    tag = "moirai-small-point"
    lr = json.load(open(FILES / "lr_moirai-small.json"))["chosen"]
    out = OUT / "family" / tag
    out.mkdir(parents=True, exist_ok=True)
    corpus, z = corpus_from_pool(seed)
    probe = np.concatenate([z["core"], z["clean"]]).astype(np.float32)

    ad = adapters.REGISTRY["moirai"]("small")
    t0 = time.time()
    ad.eval()
    before = ad.window_loss(probe)
    pb = point_losses(ad, probe)
    print(f"[{tag}] before: nll {before.mean():.4f}  "
          + "  ".join(f"{k} {pb[k].mean():.4f}" for k in POINT) + f"  ({time.time() - t0:.0f}s)",
          flush=True)

    # identical to run_family.main
    opt = torch.optim.AdamW(ad.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100))
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        ad.train()
        order = rng.permutation(len(corpus))
        tot = n = 0.0
        for i in range(0, len(order), bs):
            loss = ad.batch_loss(corpus[order[i:i + bs]])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ad.parameters(), 1.0)
            opt.step(); sch.step()
            tot += float(loss.detach()); n += 1
        print(f"[{tag}] epoch {ep + 1}/{epochs} train loss {tot / n:.4f} ({time.time() - t0:.0f}s)",
              flush=True)

    ad.eval()
    after = ad.window_loss(probe)
    pa = point_losses(ad, probe)
    torch.save(ad.net.state_dict(), out / "after.pt")

    nc = len(z["core"])
    delta = before - after
    saved = np.load(OUT / "family" / "moirai-small" / "delta.npz")["delta"]
    res = dict(family="moirai", size="small", epochs=epochs, lr=lr, batch=bs, samples=SAMPLES,
               reproduces=dict(corr=float(np.corrcoef(delta, saved)[0, 1]),
                               max_abs_diff=float(np.abs(delta - saved).max()),
                               mean_saved=float(saved.mean()), mean_now=float(delta.mean())),
               nll=analyse(delta, nc, z["core_type"], z["core_k"], epochs))
    arrays = dict(nll_before=before, nll_after=after)
    for k in POINT:
        d = pb[k] - pa[k]
        res[k] = analyse(d, nc, z["core_type"], z["core_k"], epochs)
        arrays.update({f"{k}_before": pb[k], f"{k}_after": pa[k]})
    res["seconds"] = round(time.time() - t0)
    np.savez(out / "delta.npz", n_core=nc, core_type=z["core_type"], core_k=z["core_k"], **arrays)
    json.dump(res, open(FILES / f"family_{tag}.json", "w"), indent=2)

    print(f"[{tag}] reproduces saved run: {res['reproduces']}")
    for name in ("nll",) + POINT:
        print(f"== {name}: first readable (Bonferroni) {res[name]['first_readable_exposures_bonf']}")
        for t in TYPES:
            row = " ".join(f"{res[name]['cells'][f'{t}/k{k}']['frac']:.1f}"
                           + ("*" if res[name]['cells'][f'{t}/k{k}']['readable_bonf'] else " ")
                           for k in KS)
            print(f"  {t:9} frac by k: {row}")


if __name__ == "__main__":
    main()
