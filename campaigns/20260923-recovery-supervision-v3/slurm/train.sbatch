#!/bin/bash
# Supervised training on validated recovery examples with the original BC
# anchor, from the untouched baseline. Optimisation identical to v2; a plan
# (attempt >= 2) may override single settings.
#
#   train.sbatch CAMPAIGN DATASET OUT_TAG [PLAN]
#SBATCH --job-name=lh-v3-train
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --constraint=a40|rtx8000
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

set -euo pipefail
campaign=${1:?absolute campaign root}
dataset=${2:?dataset name under datasets/}
tag=${3:?output tag}
plan=${4:-}
source "$campaign/slurm/_common.sh"
lh_runtime "train_${tag}"
M="$campaign/manifest.json"
lh_torch "$campaign/scripts/verify_sources.py" --campaign "$campaign" \
  --out "$campaign/audit/verify_sources_train_${tag}.json"
setting() {
  local key=$1 value=""
  if [ -n "$plan" ]; then value=$(jq -r --arg k "$key" '.training[$k] // empty' "$plan"); fi
  if [ -z "$value" ]; then value=$(jq -r --arg k "$key" '.training[$k]' "$M"); fi
  printf '%s' "$value"
}
init=$(jq -r '.baseline_checkpoint.path' "$M")
GRAD_ACCUM=1
if [ -n "$plan" ]; then
  init=$(jq -r '.init_checkpoint // empty' "$plan"); init=${init:-$(jq -r '.baseline_checkpoint.path' "$M")}
  # A plan that declares grad_accum must actually get it; without this wiring
  # the flag would silently default to 1 and rerun attempt 1's configuration.
  GRAD_ACCUM=$(jq -r '.training.grad_accum // 1' "$plan")
fi
lh_torch "$campaign/scripts/rollout_weighted_finetune.py" \
  --policy-path "$init" \
  --rollout-dataset "$campaign/datasets/${dataset}.npz" \
  --anchor-glob "$(jq -r '.training.anchor.glob' "$M")" \
  --anchor-files "$(jq -r '.training.anchor.files' "$M")" \
  --anchor-frames "$(jq -r '.training.anchor.frames' "$M")" \
  --heldout-glob "$(jq -r '.training.anchor_fit_metric.glob' "$M")" \
  --heldout-files "$(jq -r '.training.anchor_fit_metric.files' "$M")" \
  --heldout-frames "$(jq -r '.training.anchor_fit_metric.frames' "$M")" \
  --retention-files "$(jq -r '.training.retention.files | join(",")' "$M")" \
  --retention-frames "$(jq -r '.training.retention.frames_per_episode' "$M")" \
  --heldout-tolerance "$(jq -r '.training.retention.tolerance' "$M")" \
  --out "$campaign/training/${tag}" \
  --steps "$(setting steps)" --batch-size "$(setting batch_size)" \
  --rollout-fraction "$(setting rollout_fraction)" --lr "$(setting lr)" \
  --checkpoint-every "$(setting checkpoint_every)" --unfreeze "$(setting unfreeze)" \
  --grad-accum "$GRAD_ACCUM" \
  --seed "$(setting seed)"
