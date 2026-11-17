"""Step 5: an adaptation-dynamics signal, run on our three groups.

    python -m experiments.contamination_bound.scripts.probe --groups
    python -m experiments.contamination_bound.scripts.probe --control

THIS IS NOT A REPLICATION. The signal is the one proposed by TSFMAudit: briefly fine-tune a probe
on a candidate dataset and watch how the adaptation behaves, on the observation that contaminated
data is adapted to faster and with less movement in the backbone. Two things differ here and both
matter, so the result must be labelled as their signal on our benchmark rather than as their
method:

  UNIT.      They audit whole datasets, which is what documented corpus listings can label. Our
             groups are windows grouped by date. We follow the plan and treat each group as its
             own small dataset, which is the closest correspondence available.
  REFERENCE. They compare a target model against models that never saw the data. We have no such
             model for weather, so `clean` plays that role: the same stations as `indirect`, from
             after the checkpoint was built, so difficulty is matched by construction and only the
             period differs.

WHAT IS MEASURED. From a fixed starting checkpoint, fine-tune on a group for a short run and record
  loss_drop   the fall in mean training loss from the first steps to the last, relative to the
              starting loss -- how quickly the model adapts
  movement    ||theta_after - theta_before||_2 / ||theta_before||_2 over the tracked weights --
              how far the backbone had to move to get there
The claimed signature of contamination is a LARGE drop with a SMALL movement, so `ratio` =
loss_drop / movement is reported as the combined quantity. Every group is run from the same
starting weights, for the same number of steps, on the same number of windows, with several seeds.

THE POSITIVE CONTROL IS NOT OPTIONAL. A probe that reports nothing on all three groups is
uninterpretable on its own: it cannot be told apart from a probe that does not work in this
setting. `--control` runs the identical probe on the contaminated checkpoint from the injection
experiment, comparing windows that were injected 32 times -- provably memorised, every core above
the clean 95th percentile -- against held-out windows that were not. If the probe cannot separate
those, it cannot separate anything, and a null on the released checkpoint says nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .pool import load, FILES
from .inject import OUT, CTX, HOR, KS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tracked(net):
    """The transformer weight matrices, excluding embeddings and the output head."""
    return [(n, p) for n, p in net.named_parameters()
            if p.ndim == 2 and "shared" not in n and "lm_head" not in n]


def probe_once(model_id, win, steps, batch, lr, seed):
    """Fine-tune briefly on `win` and report how fast the loss fell and how far the weights moved."""
    from chronos import ChronosPipeline
    torch.manual_seed(seed)
    pipe = ChronosPipeline.from_pretrained(model_id, device_map=DEVICE,
                                           torch_dtype=torch.float32)
    net = pipe.model.model
    before = {n: p.detach().clone() for n, p in tracked(net)}
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    net.train()

    losses = []
    for s in range(steps):
        b = win[rng.integers(0, len(win), batch)]
        c = torch.as_tensor(b[:, :CTX])
        h = torch.as_tensor(b[:, CTX:CTX + HOR])
        ids, mask, scale = pipe.tokenizer.context_input_transform(c)
        lab, lmask = pipe.tokenizer.label_input_transform(h, scale)
        out = net(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE),
                  labels=lab.to(DEVICE).masked_fill(~lmask.to(DEVICE), -100))
        opt.zero_grad(); out.loss.backward(); opt.step()
        losses.append(float(out.loss.detach()))

    k = max(3, steps // 10)
    start, end = float(np.mean(losses[:k])), float(np.mean(losses[-k:]))
    num = sum(float(((p.detach() - before[n]) ** 2).sum()) for n, p in tracked(net))
    den = sum(float((v ** 2).sum()) for v in before.values())
    move = float(np.sqrt(num / den))
    del pipe, before
    torch.cuda.empty_cache()
    return dict(start=start, end=end, loss_drop=(start - end) / start, movement=move)


def run(model_id, sets, steps, batch, lr, seeds, n_win, tag):
    rng = np.random.default_rng(0)
    rows = {}
    for name, w in sets.items():
        out = []
        for s in range(seeds):
            sub = w[rng.choice(len(w), min(n_win, len(w)), replace=False)]
            r = probe_once(model_id, sub, steps, batch, lr, seed=s)
            r["ratio"] = r["loss_drop"] / max(r["movement"], 1e-12)
            out.append(r)
            print(f"  {name:10} seed {s}: drop {r['loss_drop']:+.4f}  "
                  f"move {r['movement']:.3e}  ratio {r['ratio']:.1f}", flush=True)
        rows[name] = out

    print(f"\n=== {tag}: {steps} steps, {n_win} windows, {seeds} seeds, mean +- sd ===")
    hdr = f"{'group':12}{'loss drop':>20}{'movement':>22}{'drop/movement':>20}"
    print(hdr); print("-" * len(hdr))
    summ = {}
    for name, out in rows.items():
        d = np.array([r["loss_drop"] for r in out])
        m = np.array([r["movement"] for r in out])
        q = np.array([r["ratio"] for r in out])
        summ[name] = dict(drop=[d.mean(), d.std(ddof=1)], move=[m.mean(), m.std(ddof=1)],
                          ratio=[q.mean(), q.std(ddof=1)], n=len(out))
        print(f"{name:12}{d.mean():>12.4f} +-{d.std(ddof=1):<7.4f}"
              f"{m.mean():>13.3e} +-{m.std(ddof=1):<8.2e}"
              f"{q.mean():>12.1f} +-{q.std(ddof=1):<7.1f}")
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", action="store_true", help="the three evaluation groups")
    ap.add_argument("--control", action="store_true", help="positive control on the injected model")
    ap.add_argument("--control-sweep", action="store_true",
                    help="run the control at EVERY injected dose, not just the strongest one.\n"
                         "The single-dose control shows only that the probe sees the most extreme\n"
                         "memorisation we could manufacture (verbatim, 32 copies, all ten cores\n"
                         "memorised). It says nothing about a faint trace. Sweeping k measures\n"
                         "where the probe's separation actually dies, which is the sensitivity\n"
                         "floor a null on the released checkpoint has to be read against.")
    ap.add_argument("--model", default="amazon/chronos-t5-small")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10, help="which injected checkpoint to use")
    ap.add_argument("--n-win", type=int, default=0,
                    help="windows per group; 0 uses the largest common size. Set this to match\n"
                         "the control, whose size is capped by the number of injected cores:\n"
                         "fewer windows means each is seen more often during the probe, which\n"
                         "makes adaptation easier, so control and groups must be compared at\n"
                         "the same size before a null on the groups means anything.")
    a = ap.parse_args()
    out = {}

    if a.groups:
        win, grp, _, _ = load()
        keep = np.load(FILES / "keep_mask.npz")["keep"]
        win, grp = win[keep], grp[keep]
        sets = {g: win[grp == g] for g in ["direct", "indirect", "clean"]}
        n = a.n_win or min(len(v) for v in sets.values())
        print(f"[groups] {[f'{k}={len(v)}' for k, v in sets.items()]}, using {n} windows each\n")
        key = "groups" if not a.n_win else f"groups_n{a.n_win}"
        out[key] = run(a.model, sets, a.steps, a.batch, a.lr, a.seeds, n, "released checkpoint")

    if a.control:
        ck = OUT / f"contaminated_ep{a.epochs}"
        z = np.load(OUT / "inject_pool.npz", allow_pickle=True)
        m = (z["core_type"] == "verbatim") & (z["core_k"] == 32)
        sets = {"injected_k32": z["core"][m], "held_out": z["clean"]}
        n = min(len(v) for v in sets.values())
        print(f"\n[control] injected k=32 verbatim n={int(m.sum())} vs held-out n={len(z['clean'])}, "
              f"using {n} each\n")
        out["control"] = run(str(ck), sets, a.steps, a.batch, a.lr, a.seeds, n,
                             "contaminated checkpoint (positive control)")

    if a.control_sweep:
        ck = OUT / f"contaminated_ep{a.epochs}"
        z = np.load(OUT / "inject_pool.npz", allow_pickle=True)
        sets = {}
        for kk in KS:
            m = (z["core_type"] == "verbatim") & (z["core_k"] == kk)
            if m.any():
                sets[f"k{kk}"] = z["core"][m]
        sets["held_out"] = z["clean"]
        n = min(len(v) for v in sets.values())
        print(f"\n[control sweep] verbatim cores at k={list(KS)} vs held-out, "
              f"{n} windows each\n")
        out["control_sweep"] = run(str(ck), sets, a.steps, a.batch, a.lr, a.seeds, n,
                                   "contaminated checkpoint, dose sweep")

    # Merge rather than overwrite: the control and the group runs are separate invocations and
    # each would otherwise erase the other's results.
    f = FILES / "probe.json"
    prev = json.load(open(f)) if f.exists() else {}
    prev.update(out)
    json.dump(prev, open(f, "w"), indent=2)
    print(f"\nwrote {FILES / 'probe.json'}")


if __name__ == "__main__":
    main()
