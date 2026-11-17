#!/bin/bash
# Reproduce the paper, stage by stage. Run from the repository root:
#
#   bash reproduce.sh tables     # every table, number and figure from the saved results (CPU, ~1 min)
#   bash reproduce.sh benchmark  # rebuild the window pool from GHCN-Daily and verify membership
#   bash reproduce.sh chronos    # score the four Chronos checkpoints, bounds and power (GPU)
#   bash reproduce.sh families   # score Moirai, TimesFM and Lag-Llama as controls (GPU)
#   bash reproduce.sh planted    # planted-leakage runs on six checkpoints (GPU, several hours)
#   bash reproduce.sh all        # the four GPU stages in order, then tables
#
# Each model family needs its own environment (see requirements/). Point these at their pythons;
# all default to the python on PATH.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}                    # base: Chronos and all analysis
PY_MOIRAI=${PY_MOIRAI:-$PY}
PY_TIMESFM=${PY_TIMESFM:-$PY}
PY_LAGLLAMA=${PY_LAGLLAMA:-$PY}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=experiments.contamination_bound.scripts

tables() {
  $PY -m $M.thresholds
  $PY -m $M.exposure_estimate
  mkdir -p out/tables out/figures
  $PY -m $M.paper_tables out/tables
  $PY -m $M.paper_figures out/figures
  echo "tables and macros in out/tables, figures in out/figures"
}

benchmark() {
  $PY -m $M.pool --build
  $PY -m $M.verify_direct --write
  $PY -m $M.export_windows benchmark_windows.csv
}

chronos() {
  # batch sizes fit a 96 GB GPU; the larger models expand to batch x 100 samples
  for spec in "mini 32" "small 32" "base 12" "large 6"; do
    set -- $spec
    $PY -m $M.score --model amazon/chronos-t5-$1 --batch $2
    for step in bound power adjust robust; do
      $PY -m $M.$step --tag chronos-t5-$1
    done
  done
  $PY -m $M.blockboot
}

families() {
  for s in mini small base large; do $PY -m $M.score_family --family chronos --size $s; done
  for s in small large; do $PY_MOIRAI -m $M.score_family --family moirai --size $s; done
  for s in 200m 500m; do $PY_TIMESFM -m $M.score_family --family timesfm --size $s; done
  $PY_LAGLLAMA -m $M.score_family --family lagllama --size base
  $PY -m $M.famcompare
}

planted() {
  # the planted corpus and the Chronos 46M run that the detectors and the probe use
  $PY -m $M.inject --build
  $PY -m $M.inject --train --epochs 10
  $PY -m $M.exposure --epochs 10
  $PY -m $M.detect
  $PY -m $M.probe --groups
  $PY -m $M.probe --control
  # learning rate per checkpoint, chosen on background windows only
  for s in small large; do $PY -m $M.lr_check --family chronos --size $s; done
  for s in small large; do $PY_MOIRAI -m $M.lr_check --family moirai --size $s; done
  for s in 200m 500m; do $PY_TIMESFM -m $M.lr_check --family timesfm --size $s; done
  $PY_LAGLLAMA -m $M.lr_check --family lagllama --size base
  # the planted runs; TimesFM at 1e-5 because its runs at the selected 3e-5 broke down part-way,
  # Moirai large is absent because no rate in the grid trained it stably
  $PY -m $M.run_family --family chronos --size small
  $PY -m $M.run_family --family chronos --size large
  $PY -m $M.run_family --family chronos --size large --per-epoch
  $PY_TIMESFM -m $M.run_family --family timesfm --size 200m --lr 1e-5 --per-epoch
  $PY_TIMESFM -m $M.run_family --family timesfm --size 500m --lr 1e-5
  $PY_MOIRAI -m $M.run_family --family moirai --size small
  $PY_MOIRAI -m $M.run_family --family moirai --size small --lr 1e-4
  $PY_LAGLLAMA -m $M.run_family --family lagllama --size base
  $PY_MOIRAI -m $M.moirai_point
}

case "${1:-tables}" in
  tables) tables ;;
  benchmark) benchmark ;;
  chronos) chronos ;;
  families) families ;;
  planted) planted ;;
  all) benchmark; chronos; families; planted; tables ;;
  *) echo "unknown stage: $1"; exit 1 ;;
esac
