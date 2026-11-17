# How much does pretraining contamination help time series foundation models?

Code and data for the paper. It measures how much a time series foundation model gains from having
seen a benchmark series, or a close relative of it, during pretraining. There are two studies:

1. **Membership benchmark (observational).** Daily rainfall windows from U.S. weather gauges,
   2010 to 2014. *Direct* gauges are in the Chronos pretraining corpus, verified value by value.
   *Indirect* gauges are nearby gauges from the same network that are not. We bound the loss gap
   between the two groups on four Chronos checkpoints, with Moirai, TimesFM and Lag-Llama as
   controls.
2. **Planted leakage (interventional).** Windows are planted in a training corpus 1 to 32 times,
   as exact copies, near-duplicates or blends with other series. Six checkpoints are trained on it
   for ten epochs to find how many exposures it takes before a planted window is memorised.

## Quick start: the paper's tables and figures

The saved results are in the repository, so every table, every number quoted in the text and every
figure can be rebuilt on a CPU in about a minute:

```bash
pip install -r requirements/base.txt
bash reproduce.sh tables
```

This writes the LaTeX tables and the number macros to `out/tables/` and the figures to
`out/figures/`. They are the files the paper is compiled from.

## The two supplementary files

- **`benchmark_windows.csv`**: every evaluation window, one per row: GHCN station id, group
  (`direct`, `indirect` or `clean`), first and last day of the 576-day window, and whether it is
  kept. One direct window could not be matched value by value and is excluded from every result.
- **`experiments/contamination_bound/scripts/exposure_estimate.py`**: how often Chronos
  pretraining shows a given USHCN window, from the published training setup and the four
  assumptions stated in its docstring. Needs only numpy:

  ```bash
  python -m experiments.contamination_bound.scripts.exposure_estimate
  ```

## Running everything from scratch

`reproduce.sh` runs the pipeline in stages, each overwriting the saved results it produces:

| stage | what it does | needs |
|---|---|---|
| `benchmark` | download GHCN-Daily, build the window pool, verify membership against the Chronos corpus | internet, CPU |
| `chronos` | score the four Chronos checkpoints; raw bound, difficulty adjustment, robustness, power, block bootstrap | GPU |
| `families` | score Moirai, TimesFM and Lag-Llama on the same windows; cross-family control | GPU, all four environments |
| `planted` | build the planted corpus, choose learning rates, train and score six checkpoints, Moirai point-loss check | GPU, all four environments, several hours |
| `tables` | thresholds, exposure estimate, tables, figures | CPU |

`bash reproduce.sh all` runs the four GPU stages and then `tables`. All results in the paper used
one 96 GB GPU. Training and bootstrap seeds are fixed.

### Environments

The four model families pin conflicting dependencies, so each gets its own environment, built on
top of the base one:

```bash
python3.10 -m venv base && base/bin/pip install -r requirements/base.txt
for fam in moirai timesfm lagllama; do
  base/bin/python -m venv --system-site-packages $fam
done
moirai/bin/pip install --no-deps uni2ts==2.0.0 && moirai/bin/pip install -r requirements/moirai.txt
timesfm/bin/pip install --no-deps -r requirements/timesfm.txt
lagllama/bin/pip install --no-deps "lag-llama @ git+https://github.com/time-series-foundation-models/lag-llama.git"
lagllama/bin/pip install "gluonts[torch]==0.14.4" lightning==2.1.4

PY=base/bin/python PY_MOIRAI=moirai/bin/python PY_TIMESFM=timesfm/bin/python \
PY_LAGLLAMA=lagllama/bin/python bash reproduce.sh all
```

### Data

- Rainfall: NOAA GHCN-Daily, `https://www.ncei.noaa.gov/pub/data/ghcn/daily/`. Downloaded on first
  use into `$GHCN_CACHE` (default `ghcn_cache/`).
- Chronos corpus: `ushcn_daily` in the Hugging Face dataset `autogluon/chronos_datasets`.
- Model checkpoints are downloaded from the Hugging Face Hub on first use.

## Layout

```
reproduce.sh                       the pipeline, stage by stage
benchmark_windows.csv              supplementary file 1
requirements/                      one file per environment
experiments/contamination_bound/
  scripts/                         one module per step; each docstring says what it does and why
  files/                           saved results the tables are built from (JSON, per-window NPZ)
outputs/                           planted corpus and per-window loss changes of the planted runs
                                   ($CB_OUT; trained weights are written here too)
```

| step | scripts |
|---|---|
| window pool, membership verification | `pool.py`, `verify_direct.py`, `export_windows.py` |
| scoring | `adapters.py` (one class per family), `score.py`, `score_family.py` |
| bound | `bound.py`, `adjust.py`, `robust.py`, `power.py`, `blockboot.py`, `famcompare.py` |
| detectors and adaptation probe | `detect.py`, `probe.py` |
| planted leakage | `inject.py`, `exposure.py`, `lr_check.py`, `run_family.py`, `moirai_point.py`, `thresholds.py` |
| pretraining exposure | `exposure_estimate.py` |
| paper output | `paper_tables.py`, `paper_figures.py`, `summarize.py` |
