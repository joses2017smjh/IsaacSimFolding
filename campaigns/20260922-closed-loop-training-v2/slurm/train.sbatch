#!/bin/bash
# Fixed-seed AWR-resampled rollout + raster-BC-anchor update. Iteration 1 uses
# the frozen manifest; later iterations pass a resolved plan, which supplies the
# initial checkpoint and any single changed training factor. Everything else
# comes from the manifest.
#
#   train.sbatch CAMPAIGN SCOPE [PLAN]
#SBATCH --job-name=lh-cl2-train
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --constraint=a40|rtx8000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

set -euo pipefail
campaign=${1:?absolute campaign root}
scope=${2:-iteration1}
plan=${3:-}
source "$campaign/slurm/_common.sh"
lh_runtime "train_${scope}"
M="$campaign/manifest.json"

gate="$campaign/audit/collection_gate_${scope}.json"
if [ "$(jq -r '.passed' "$gate")" != "true" ]; then
  echo "collection gate did not pass: $(jq -c '.unmet_requirements' "$gate")" >&2
  exit 4
fi
lh_torch "$campaign/scripts/verify_sources.py" --campaign "$campaign" \
  --out "$campaign/audit/verify_sources_train_${scope}.json"

# A plan value wins where present; the manifest supplies every default.
setting() {
  local key=$1 value=""
  if [ -n "$plan" ]; then value=$(jq -r --arg k "$key" '.training[$k] // empty' "$plan"); fi
  if [ -z "$value" ]; then value=$(jq -r --arg k "$key" '.training[$k]' "$M"); fi
  printf '%s' "$value"
}
init=$(jq -r '.baseline_checkpoint.path' "$M")
if [ -n "$plan" ]; then init=$(jq -r '.init_checkpoint' "$plan"); fi

lh_torch "$campaign/scripts/rollout_weighted_finetune.py" \
  --policy-path "$init" \
  --rollout-dataset "$campaign/datasets/${scope}.npz" \
  --anchor-glob "$(jq -r '.training.anchor.glob' "$M")" \
  --anchor-files "$(jq -r '.training.anchor.files' "$M")" \
  --anchor-frames "$(jq -r '.training.anchor.frames' "$M")" \
  --heldout-glob "$(jq -r '.training.heldout.glob' "$M")" \
  --heldout-files "$(jq -r '.training.heldout.files' "$M")" \
  --heldout-frames "$(jq -r '.training.heldout.frames' "$M")" \
  --heldout-tolerance "$(jq -r '.training.heldout.tolerance' "$M")" \
  --out "$campaign/training/${scope}" \
  --steps "$(setting steps)" \
  --batch-size "$(setting batch_size)" \
  --rollout-fraction "$(setting rollout_fraction)" \
  --lr "$(setting lr)" \
  --checkpoint-every "$(setting checkpoint_every)" \
  --unfreeze "$(setting unfreeze)" \
  --seed "$(setting seed)"
