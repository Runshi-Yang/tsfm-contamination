"""One adapter per model family, all with the same small interface.

    load via REGISTRY[family](size)
    window_loss(win)  -> per-window loss on the 64-step horizon, no grad   (np.ndarray [N])
    batch_loss(win)   -> mean training loss over a batch, with grad        (torch scalar)
    parameters(), train(), eval(), n_params(), LOSS (name of the objective)

`win` is float32 [N, 576]: 512 days of context then 64 days of horizon.

Each family's package lives in its own environment (their pinned dependencies conflict), so the
imports are inside the classes; importing this module never pulls in a family you are not running.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

CTX, HOR = 512, 64
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class _Base:
    LOSS = "?"
    net: torch.nn.Module

    def parameters(self):
        return [p for p in self.net.parameters() if p.requires_grad]

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def n_params(self):
        return int(sum(p.numel() for p in self.net.parameters()))

    @torch.no_grad()
    def window_loss(self, win, batch=32):
        return np.concatenate([self._loss(win[i:i + batch], reduce=False).float().cpu().numpy()
                               for i in range(0, len(win), batch)])

    def batch_loss(self, win):
        return self._loss(win, reduce=True)


# ------------------------------------------------------------------------------------ Chronos

class Chronos(_Base):
    """Token cross-entropy summed over the 65 horizon tokens (64 steps + EOS), as in score.py."""
    LOSS = "token cross-entropy (summed nats)"
    SIZES = {"mini": "amazon/chronos-t5-mini", "small": "amazon/chronos-t5-small",
             "base": "amazon/chronos-t5-base", "large": "amazon/chronos-t5-large"}

    def __init__(self, size):
        from chronos import ChronosPipeline
        self.pipe = ChronosPipeline.from_pretrained(self.SIZES[size], device_map=DEV,
                                                    torch_dtype=torch.float32)
        self.net = self.pipe.model.model

    def _loss(self, win, reduce):
        c, h = torch.as_tensor(win[:, :CTX]), torch.as_tensor(win[:, CTX:CTX + HOR])
        ids, mask, scale = self.pipe.tokenizer.context_input_transform(c)
        lab, lmask = self.pipe.tokenizer.label_input_transform(h, scale)
        ids, mask, lab, lmask = ids.to(DEV), mask.to(DEV), lab.to(DEV), lmask.to(DEV)
        o = self.net(input_ids=ids, attention_mask=mask, labels=lab.masked_fill(~lmask, -100))
        if reduce:
            return o.loss
        ce = F.cross_entropy(o.logits.transpose(1, 2), lab, reduction="none")
        return (ce * lmask).sum(-1)


# ------------------------------------------------------------------------------------- Moirai

class Moirai(_Base):
    """Negative log-likelihood of the horizon under Moirai's mixture distribution, from uni2ts's
    own per-sample validation loss. Patch size is fixed at 32 so it divides both 512 and 64;
    'auto' would pick the patch size by validation loss on each call, which would let the
    measurement change its own settings between the before and after passes."""
    LOSS = "negative log-likelihood (per-sample, uni2ts SampleNLLLoss)"
    SIZES = {"small": "Salesforce/moirai-1.0-R-small", "base": "Salesforce/moirai-1.0-R-base",
             "large": "Salesforce/moirai-1.0-R-large"}
    PATCH = 32

    def __init__(self, size):
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        mod = MoiraiModule.from_pretrained(self.SIZES[size])
        self.fc = MoiraiForecast(module=mod, prediction_length=HOR, context_length=CTX,
                                 patch_size=self.PATCH, num_samples=100, target_dim=1,
                                 feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).to(DEV)
        self.net = self.fc.module

    def _loss(self, win, reduce):
        x = torch.as_tensor(win, device=DEV).unsqueeze(-1)            # [B, 576, 1]
        obs = torch.ones_like(x, dtype=torch.bool)
        pad = torch.zeros(x.shape[:2], dtype=torch.bool, device=DEV)
        nll = self.fc._val_loss(self.PATCH, target=x, observed_target=obs, is_pad=pad)
        return nll.mean() if reduce else nll


# ------------------------------------------------------------------------------------ TimesFM

class TimesFM(_Base):
    """TimesFM has no likelihood; it is trained on squared error of its mean head plus the pinball
    loss of its quantile heads, which is what we use. Both are computed on values divided by the
    context's standard deviation, so a wet window does not count for more than a dry one merely
    because its numbers are larger. Needs timesfm==1.x, the last version that loads the 1.0 and
    2.0 PyTorch checkpoints."""
    LOSS = "squared error + pinball loss over 9 deciles, context-standardised"
    SIZES = {"200m": ("google/timesfm-1.0-200m-pytorch", 20, True),
             "500m": ("google/timesfm-2.0-500m-pytorch", 50, False)}

    def __init__(self, size):
        from huggingface_hub import snapshot_download
        from timesfm import pytorch_patched_decoder as ppd
        repo, layers, pos = self.SIZES[size]
        cfg = ppd.TimesFMConfig(num_layers=layers, num_heads=16, hidden_size=1280,
                                intermediate_size=1280, head_dim=80, patch_len=32,
                                horizon_len=128, use_positional_embedding=pos)
        self.net = ppd.PatchedTimeSeriesDecoder(cfg)
        ck = torch.load(f"{snapshot_download(repo)}/torch_model.ckpt", weights_only=True)
        self.net.load_state_dict(ck)
        self.net.to(DEV)
        self.q = torch.tensor(cfg.quantiles, device=DEV)

    def _loss(self, win, reduce):
        x = torch.as_tensor(win, device=DEV)
        ctx, y = x[:, :CTX], x[:, CTX:CTX + HOR]
        out = self.net(ctx, torch.zeros_like(ctx), torch.zeros(len(x), 1, dtype=torch.long,
                                                               device=DEV))
        out = out[:, -1, :HOR, :]                                   # [B, 64, 1 + 9]
        s = ctx.std(1, keepdim=True) + 1.0
        yn, mean, qs = y / s, out[..., 0] / s, out[..., 1:] / s.unsqueeze(-1)
        err = yn.unsqueeze(-1) - qs
        pin = torch.maximum(self.q * err, (self.q - 1) * err).mean((-1, -2))
        loss = ((mean - yn) ** 2).mean(-1) + pin
        return loss.mean() if reduce else loss


# ---------------------------------------------------------------------------------- Lag-Llama

class LagLlama(_Base):
    """Student-t negative log-likelihood of the 64 horizon steps (Lag-Llama also scores its 31
    context positions during pretraining; we keep only the horizon, as for every other family).

    Two consequences of its design. It reads lagged values up to 1092 steps back, more than our
    512-day context, so the missing front is zero-padded and marked unobserved -- the same thing
    gluonts does for any short series. And it was pretrained with calendar features, which need
    dates the planted pool does not record; every window is therefore given the same nominal start
    date. The features are identical before and after fine-tuning, so they cancel in delta.
    Only one checkpoint size has been released."""
    LOSS = "Student-t negative log-likelihood of the horizon"
    SIZES = {"base": ("time-series-foundation-models/Lag-Llama", "lag-llama.ckpt")}

    def __init__(self, size):
        import pandas as pd
        from gluonts.time_feature import time_features_from_frequency_str
        from huggingface_hub import hf_hub_download
        from lag_llama.gluon.lightning_module import LagLlamaLightningModule
        p = hf_hub_download(*self.SIZES[size])
        import os
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"   # official checkpoint, pickled gluonts objects
        self.lm = LagLlamaLightningModule.load_from_checkpoint(p, map_location=DEV)
        self.lm.to(DEV)
        self.net = self.lm.model
        self.past = self.lm.context_length + max(self.net.lags_seq)
        idx = pd.date_range("2012-01-01", periods=self.past + HOR, freq="D")
        tf = np.stack([f(idx) for f in time_features_from_frequency_str("S")], -1)
        self.tf = torch.as_tensor(tf, dtype=torch.float32, device=DEV)

    def _loss(self, win, reduce):
        x = torch.as_tensor(win, device=DEV)
        b, pad = len(x), self.past - CTX
        past = torch.cat([torch.zeros(b, pad, device=DEV), x[:, :CTX]], 1)
        obs = torch.cat([torch.zeros(b, pad, device=DEV), torch.ones(b, CTX, device=DEV)], 1)
        batch = dict(past_target=past, past_observed_values=obs,
                     future_target=x[:, CTX:CTX + HOR],
                     future_observed_values=torch.ones(b, HOR, device=DEV),
                     past_time_feat=self.tf[:self.past].expand(b, -1, -1),
                     future_time_feat=self.tf[self.past:].expand(b, -1, -1))
        nll = self.lm._compute_loss(batch, do_not_average=True)[:, -HOR:].mean(-1)
        return nll.mean() if reduce else nll


REGISTRY = {"chronos": Chronos, "moirai": Moirai, "timesfm": TimesFM, "lagllama": LagLlama}
