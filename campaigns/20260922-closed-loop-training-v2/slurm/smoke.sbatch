#!/bin/bash
# End-to-end launch gate. Proves the pipeline connects:
#   verify sources -> one short Isaac rollout -> persisted aligned trajectory
#   -> compiled dataset -> one real optimizer update -> checkpoint saved
#   -> checkpoint reloaded and producing a finite action.
#
# This is NOT a policy-quality measurement. One episode cannot produce a
# reward spread, so the collection gate is recorded with --allow-degenerate
# and deliberately ignored here; the production compile enforces it.
#SBATCH --job-name=lh-cl2-smoke
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --constraint=a40|rtx8000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail
campaign=${1:?absolute campaign root}
source "$campaign/slurm/_common.sh"
lh_runtime "smoke"
M="$campaign/manifest.json"

echo "=== [1/5] verify executed sources against the frozen manifest ==="
lh_torch "$campaign/scripts/verify_sources.py" --campaign "$campaign" \
  --out "$campaign/audit/verify_sources_smoke.json"

echo "=== [2/5] one 80-action autonomous H10 rollout ==="
lh_isaac "$campaign/scripts/run_rollout_task.py" \
  --campaign "$campaign" --phase smoke --index 0 --label smoke-baseline

echo "=== [3/5] compile the trajectory into aligned AWR samples ==="
lh_torch "$campaign/scripts/build_rollout_dataset.py" \
  --trajectory-glob "$campaign/smoke/rollouts/*/trajectory.npz" \
  --manifest "$M" \
  --out "$campaign/smoke/dataset.npz" \
  --provenance "$campaign/smoke/dataset-provenance.json" \
  --csv "$campaign/smoke/dataset-episodes.csv" \
  --gate-out "$campaign/smoke/collection_gate.json" \
  --allow-degenerate

echo "=== [4/5] one mixed AWR-resampled update, checkpoint saved ==="
lh_torch "$campaign/scripts/rollout_weighted_finetune.py" \
  --policy-path "$(jq -r '.baseline_checkpoint.path' "$M")" \
  --rollout-dataset "$campaign/smoke/dataset.npz" \
  --anchor-glob "$(jq -r '.training.anchor.glob' "$M")" \
  --anchor-files 2 --anchor-frames 2 \
  --heldout-glob "$(jq -r '.training.heldout.glob' "$M")" \
  --heldout-files 2 --heldout-frames 2 \
  --out "$campaign/smoke/training" --steps 1 --batch-size 2 \
  --rollout-fraction 0.5 --checkpoint-every 1 \
  --unfreeze "$(jq -r '.training.unfreeze' "$M")" --seed 4242

echo "=== [5/5] reload the saved checkpoint as the evaluator does ==="
lh_torch "$campaign/scripts/check_checkpoint.py" \
  --checkpoint "$campaign/smoke/training/checkpoints/step_000001" \
  --dataset "$campaign/smoke/dataset.npz" \
  --baseline "$(jq -r '.baseline_checkpoint.path' "$M")" \
  --out "$campaign/smoke/checkpoint-load.json"

echo "=== SMOKE COMPLETE ==="
