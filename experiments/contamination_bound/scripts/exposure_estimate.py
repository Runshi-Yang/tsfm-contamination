"""How often did Chronos pretraining show the model a given USHCN window? An order of magnitude.

    python -m experiments.contamination_bound.scripts.exposure_estimate

WHY. The paper compares planted doses with "about one exposure in pretraining". That number has to
be derived, and the unit has to match. A planted copy is an exact 576-day window presented whole.
Chronos pretraining instead slices random 576-step crops out of TSMixup augmentations, so a given
window is rarely presented exactly but its days appear inside overlapping crops. Both are computed.

INPUTS, all from Ansari et al. (2024), Section 5.2, Table 3 and Algorithm 1:
  200K steps x 256 sequences; sequences are random 576-step windows (512 context + 64 horizon)
  sliced from series drawn 9:1 from 10M TSMixup augmentations and 1M KernelSynth series.
  TSMixup: k ~ U{1,2,3} series mixed; length l ~ U{128..2048}; each component's dataset drawn
  uniformly from the 28 training datasets; convex Dirichlet(1.5) weights after mean scaling.
  USHCN: 6,090 series (1,218 stations x 5 variables), mean length 38,653 days.

ASSUMPTIONS (not stated in the paper, needed for a number):
  A1  within a dataset, the series and the start of the length-l segment are drawn uniformly;
  A2  training draws are uniform over the 10M augmentations;
  A3  a training crop of an augmentation shorter than 576 steps contains all of it (padding);
  A4  all five USHCN variables are treated alike, so precipitation days share the USHCN rate.
"""
from __future__ import annotations

import json

import numpy as np

from .pool import FILES

STEPS, BATCH, AUG_SHARE = 200_000, 256, 0.9
N_AUG, K, L_MIN, L_MAX, N_DATASETS = 10_000_000, 3, 128, 2048, 28
WIN = 576
USHCN_SERIES, USHCN_MEAN_LEN = 6_090, 38_653


def main():
    draws_per_aug = STEPS * BATCH * AUG_SHARE / N_AUG
    ls = np.arange(L_MIN, L_MAX + 1)
    p_k = {k: 1 / K for k in range(1, K + 1)}
    unmixed_aug = N_AUG * p_k[1] / N_DATASETS                       # k = 1 and the dataset is USHCN
    mixed_comp = N_AUG * sum(p_k[k] * k for k in (2, 3)) / N_DATASETS  # USHCN components inside blends
    ushcn_days = USHCN_SERIES * USHCN_MEAN_LEN
    ushcn_windows = USHCN_SERIES * (USHCN_MEAN_LEN - WIN + 1)
    # expected training crops covering one particular day of one augmentation, averaged over l
    cover = np.minimum(ls, WIN).mean()                             # = E_l[ l * min(1, WIN/l) ]
    day_unmixed = unmixed_aug / ushcn_days * draws_per_aug * cover
    day_in_blends = mixed_comp / ushcn_days * draws_per_aug * cover
    # exact presentation: an unmixed augmentation long enough to hold the window, and a crop that
    # starts exactly at the window's first day. For l >= WIN the l-dependence cancels.
    p_long = (ls >= WIN).mean()
    exact = unmixed_aug * p_long * draws_per_aug / ushcn_windows
    out = dict(draws_per_augmentation=draws_per_aug, unmixed_ushcn_augmentations=unmixed_aug,
               day_exposures_unmixed=day_unmixed, day_exposures_in_blends=day_in_blends,
               exact_window_exposures=exact, one_in=1 / exact)
    for k, v in out.items():
        print(f"{k:34} {v:,.4g}")
    json.dump(out, open(FILES / "exposure_estimate.json", "w"), indent=2)


if __name__ == "__main__":
    main()
