#!/bin/bash
# v3 launch gate: every NEW component once, before any production task.
#   verify sources -> short recovery search (runner mode, outputs, restore)
#   -> coverage compiler on it (gate verdict recorded) -> one training step
#   with the new whole-episode retention guard -> reload as the evaluator does.
# Training uses v2's already-compiled iteration-2 dataset purely as plumbing:
# a 120-action search cannot be expected to produce settled successes.
#SBATCH --job-name=lh-v3-smoke
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --constraint=a40|rtx8000
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail
campaign=${1:?absolute campaign root}
source "$campaign/slurm/_common.sh"
lh_runtime "smoke"
M="$campaign/manifest.json"
S="$campaign/smoke"
echo "=== [1/5] verify sources ==="
lh_torch "$campaign/scripts/verify_sources.py" --campaign "$campaign" --out "$campaign/audit/verify_sources_smoke.json"
echo "=== [2/5] short recovery search ==="
lh_isaac "$campaign/scripts/run_rollout_task.py" --campaign "$campaign" --phase recovery_smoke --index 0 --label smoke
echo "=== [3/5] coverage compiler ==="
lh_torch "$campaign/scripts/build_recovery_dataset.py" --campaign "$campaign" --smoke \
  --out "$S/recovery.npz" --provenance "$S/recovery-provenance.json" --gate-out "$S/coverage_gate.json" || \
  { rc=$?; [ "$rc" -eq 4 ] && echo "coverage gate unmet on the smoke search, as expected" || exit "$rc"; }
echo "=== [4/5] one training step with the retention guard ==="
lh_torch "$campaign/scripts/rollout_weighted_finetune.py" \
  --policy-path "$(jq -r '.baseline_checkpoint.path' "$M")" \
  --rollout-dataset "$campaign/../20260922-closed-loop-training-v2/datasets/iteration2.npz" \
  --anchor-glob "$(jq -r '.training.anchor.glob' "$M")" --anchor-files 4 --anchor-frames 12 \
  --heldout-glob "$(jq -r '.training.anchor_fit_metric.glob' "$M")" --heldout-files 2 --heldout-frames 2 \
  --retention-files "$(jq -r '.training.retention.files | join(",")' "$M")" --retention-frames 4 \
  --out "$S/training" --steps 1 --batch-size 4 --grad-accum 8 --anchor-mode whole_episode \
  --rollout-fraction 0.5 --checkpoint-every 1 --unfreeze boundary --seed 4242
echo "=== [5/5] reload ==="
lh_torch "$campaign/scripts/check_checkpoint.py" --checkpoint "$S/training/checkpoints/step_000001" \
  --dataset "$campaign/../20260922-closed-loop-training-v2/datasets/iteration2.npz" \
  --baseline "$(jq -r '.baseline_checkpoint.path' "$M")" --out "$S/checkpoint-load.json"
echo "=== SMOKE COMPLETE ==="
