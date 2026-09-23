#!/bin/bash
# One fixed-seed 300-step 50/50 AWR-resampled rollout + raster-BC-anchor update
# from the untouched baseline. Every constant comes from the frozen manifest.
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

lh_torch "$campaign/scripts/rollout_weighted_finetune.py" \
  --policy-path "$(jq -r '.baseline_checkpoint.path' "$M")" \
  --rollout-dataset "$campaign/datasets/${scope}.npz" \
  --anchor-glob "$(jq -r '.training.anchor.glob' "$M")" \
  --anchor-files "$(jq -r '.training.anchor.files' "$M")" \
  --anchor-frames "$(jq -r '.training.anchor.frames' "$M")" \
  --heldout-glob "$(jq -r '.training.heldout.glob' "$M")" \
  --heldout-files "$(jq -r '.training.heldout.files' "$M")" \
  --heldout-frames "$(jq -r '.training.heldout.frames' "$M")" \
  --heldout-tolerance "$(jq -r '.training.heldout.tolerance' "$M")" \
  --out "$campaign/training/${scope}" \
  --steps "$(jq -r '.training.steps' "$M")" \
  --batch-size "$(jq -r '.training.batch_size' "$M")" \
  --rollout-fraction "$(jq -r '.training.rollout_fraction' "$M")" \
  --lr "$(jq -r '.training.lr' "$M")" \
  --checkpoint-every "$(jq -r '.training.checkpoint_every' "$M")" \
  --unfreeze "$(jq -r '.training.unfreeze' "$M")" \
  --seed "$(jq -r '.training.seed' "$M")"
