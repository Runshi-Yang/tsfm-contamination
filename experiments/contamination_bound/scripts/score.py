"""Forecast error per window, in three metrics, for the three-group pool.

    python -m experiments.contamination_bound.scripts.score --model amazon/chronos-t5-small

Each 576-day window is split into 512 days of context and a 64-day horizon, which is
exactly chronos-t5's `context_length` / `prediction_length`, so every window is scored the
way the model was trained.

METRICS.
  ce      Token cross-entropy summed over the 64 horizon tokens -- Chronos's own training
          objective, computed by tokenising the window and teacher-forcing the horizon.
          `ce_mean` is the same quantity per token.
  MASE    mean |y - yhat| over the horizon, divided by the mean |Y_t - Y_{t-m}| of the
          CONTEXT. Point forecast is the median sample, as in the Chronos paper. Reported
          at m=1 (daily precipitation has no reliable weekly cycle) with m=7 alongside as
          a robustness check.
  WQL     Weighted quantile loss over the 9 deciles, normalised by sum |y| on the horizon.

MAPE is deliberately absent: precipitation has too many zero days for a ratio to exist.

TWO GUARDS, BOTH REPORTED PER GROUP. MASE is undefined when the context never changes
(an all-dry context makes the seasonal-naive denominator zero) and WQL is undefined when
the horizon is entirely dry (the normaliser is zero). Windows failing a guard are dropped
from THAT metric only. The drop rate is printed per group because a rate that differs
across direct/indirect/clean would itself be a confound -- the surviving windows would no
longer be comparable groups.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .pool import load

FILES = Path(__file__).resolve().parents[1] / "files"
QUANTILES = np.arange(1, 10) / 10.0
CTX, HOR = 512, 64


def token_ce(pipe, ctx, hor, device):
    """Summed and per-token horizon cross-entropy, teacher-forced."""
    tok = pipe.tokenizer
    ids, mask, scale = tok.context_input_transform(torch.as_tensor(ctx))
    lab, lmask = tok.label_input_transform(torch.as_tensor(hor), scale)
    ids, mask = ids.to(device), mask.to(device)
    lab, lmask = lab.to(device), lmask.to(device)
    out = pipe.model.model(input_ids=ids, attention_mask=mask,
                           labels=lab.masked_fill(~lmask, -100))
    ce = F.cross_entropy(out.logits.transpose(1, 2), lab, reduction="none")
    ce = ce * lmask
    return ce.sum(-1).float().cpu().numpy(), lmask.sum(-1).float().cpu().numpy()


def metrics_from_samples(samples, ctx, hor):
    """samples: (B, S, HOR) forecast paths on the original scale."""
    med = np.median(samples, axis=1)
    mae = np.abs(hor - med).mean(1)

    den = {}
    for m in (1, 7):
        den[m] = np.abs(ctx[:, m:] - ctx[:, :-m]).mean(1)

    q = np.quantile(samples, QUANTILES, axis=1)               # (Q, B, HOR)
    y = hor[None]
    ql = 2.0 * (QUANTILES[:, None, None] * np.clip(y - q, 0, None)
                + (1 - QUANTILES)[:, None, None] * np.clip(q - y, 0, None))
    wql_num = ql.sum(2).mean(0)                               # mean over quantiles
    wql_den = np.abs(hor).sum(1)

    # CRPS by the energy-form sample estimator, E|X-y| - 0.5 E|X-X'|. WQL as the Chronos
    # paper defines it divides by sum|y|, which is degenerate for a zero-inflated variable
    # like daily precipitation: predicting zero everywhere scores exactly 1.0. Scaling CRPS
    # by the same seasonal-naive denominator MASE uses keeps a probabilistic metric that
    # still discriminates here.
    s1 = np.abs(samples - hor[:, None, :]).mean(1).mean(1)
    sub = samples[:, ::5, :]                                  # subsample: the pair term is
    d = np.abs(sub[:, :, None, :] - sub[:, None, :, :])       # O(S^2) and converges fast
    s2 = d.mean(axis=(1, 2)).mean(1)
    crps = s1 - 0.5 * s2
    return mae, den, wql_num, wql_den, crps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="amazon/chronos-t5-small")
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or a.model.split("/")[-1]

    from chronos import ChronosPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = ChronosPipeline.from_pretrained(a.model, device_map=device,
                                           torch_dtype=torch.float32)
    torch.manual_seed(0)

    win, grp, sta, corr = load()
    ctx, hor = win[:, :CTX].astype(np.float64), win[:, CTX:CTX + HOR].astype(np.float64)
    print(f"[pool] {len(win)} windows | " +
          " ".join(f"{g}={int((grp == g).sum())}" for g in ["direct", "indirect", "clean"]))

    ce_sum, ce_ntok, mae, wql_n, wql_d, crps_l = [], [], [], [], [], []
    den1, den7 = [], []
    for i in range(0, len(win), a.batch):
        cb = torch.tensor(win[i:i + a.batch, :CTX], dtype=torch.float32)
        hb = torch.tensor(win[i:i + a.batch, CTX:CTX + HOR], dtype=torch.float32)
        with torch.no_grad():
            s, n = token_ce(pipe, cb, hb, device)
            samp = pipe.predict(cb, prediction_length=HOR, num_samples=a.num_samples,
                                limit_prediction_length=False)
        samp = samp.float().cpu().numpy()
        m, d, wn, wd, cr = metrics_from_samples(samp, ctx[i:i + a.batch], hor[i:i + a.batch])
        ce_sum.append(s); ce_ntok.append(n); mae.append(m); crps_l.append(cr)
        den1.append(d[1]); den7.append(d[7]); wql_n.append(wn); wql_d.append(wd)
        if (i // a.batch) % 20 == 0:
            print(f"  scored {i + len(s):5}/{len(win)}", flush=True)

    ce_sum = np.concatenate(ce_sum); ce_ntok = np.concatenate(ce_ntok)
    mae = np.concatenate(mae); den1 = np.concatenate(den1); den7 = np.concatenate(den7)
    wql_n = np.concatenate(wql_n); wql_d = np.concatenate(wql_d)
    crps = np.concatenate(crps_l)

    eps = 1e-9
    ok1, ok7, okw = den1 > eps, den7 > eps, wql_d > eps
    mase1 = np.where(ok1, mae / np.where(ok1, den1, 1), np.nan)
    mase7 = np.where(ok7, mae / np.where(ok7, den7, 1), np.nan)
    wql = np.where(okw, wql_n / np.where(okw, wql_d, 1), np.nan)
    scrps = np.where(ok1, crps / np.where(ok1, den1, 1), np.nan)

    np.savez(FILES / f"scores_{tag}.npz", group=grp, station=sta, corpus_corr=corr,
             ce_sum=ce_sum, ce_mean=ce_sum / np.maximum(ce_ntok, 1), mae=mae,
             mase1=mase1, mase7=mase7, wql=wql, scrps=scrps, crps=crps,
             dry_ctx=(win[:, :CTX] <= 0).mean(1), dry_hor=(hor <= 0).mean(1))

    print(f"\n=== {tag}: mean error by group (n = usable windows) ===")
    hdr = f"{'group':10}{'n':>6}{'ce_sum':>10}{'ce_mean':>9}{'MASE m=1':>10}{'MASE m=7':>10}{'WQL':>9}{'sCRPS':>9}"
    print(hdr); print("-" * len(hdr))
    summary = {}
    for g in ["direct", "indirect", "clean"]:
        m = grp == g
        row = dict(n=int(m.sum()),
                   ce_sum=float(ce_sum[m].mean()),
                   ce_mean=float((ce_sum / np.maximum(ce_ntok, 1))[m].mean()),
                   mase1=float(np.nanmean(mase1[m])), mase7=float(np.nanmean(mase7[m])),
                   wql=float(np.nanmean(wql[m])), scrps=float(np.nanmean(scrps[m])),
                   drop_mase1=float((~ok1[m]).mean()), drop_wql=float((~okw[m]).mean()),
                   dry_hor=float((hor[m] <= 0).mean()))
        summary[g] = row
        print(f"{g:10}{row['n']:>6}{row['ce_sum']:>10.2f}{row['ce_mean']:>9.3f}"
              f"{row['mase1']:>10.3f}{row['mase7']:>10.3f}{row['wql']:>9.4f}{row['scrps']:>9.4f}")
    print("\nguard drop rates (a rate that differs across groups is itself a confound)")
    for g in ["direct", "indirect", "clean"]:
        print(f"  {g:10} MASE dropped {summary[g]['drop_mase1']:.3%}   "
              f"WQL dropped {summary[g]['drop_wql']:.3%}   "
              f"dry-day frac (horizon) {summary[g]['dry_hor']:.3f}")
    json.dump(summary, open(FILES / f"scores_{tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
