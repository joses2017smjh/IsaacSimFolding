#!/bin/bash
#SBATCH --job-name=lh-evalprobe
#SBATCH --account=eecs
#SBATCH --partition=share,gpu,ampere,dgxh
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Does the OFFICIAL evaluator run on this stack at all?
#
# Stages 3 and 4 both route through scripts/run_eval.py, which hands control to
# LeHome's own scripts.eval with --enable_cameras. That is the Isaac Lab render
# product, and 5.1's RTX delegate segfaults against this driver -- which is why
# every rollout in this repo goes through Storm instead. Reading the code says
# this cannot work; running it is what makes that a finding rather than a
# guess.
#
# Exit code is not the signal: Kit swallows sys.exit, so this greps stdout.

set -euo pipefail
export LH_ROUTE=${LH_ROUTE:-source}
source /nfs/hpc/share/$USER/Humanoid_Lite/lehome-fold-repro/slurm/_env.sh
setup_node_cache
echo "=== node: $(hostname) | $(date) ==="
LOG=$WORKSPACE/logs/evalprobe-${SLURM_JOB_ID}.txt
set +e
lh_exec bash -c 'cd "$LEHOME" && "$PY" -u "$REPO/scripts/run_eval.py" \
    --policy_type lerobot \
    --policy_path "'"$POLICY_PATH"'" \
    --dataset_root Datasets/example/four_types_merged \
    --garment_type pant_short --num_episodes 1 --max_steps 60 \
    --device cpu --enable_cameras --headless' 2>&1 | tee "$LOG"
set -e
echo "--- verdict ---"
if grep -qiE "Segmentation fault|librtx|core dumped|Fatal" "$LOG"; then
  echo "OFFICIAL EVAL: crashed (RTX path) -- Stages 3/4 blocked as predicted"
elif grep -qiE "episode|success" "$LOG"; then
  echo "OFFICIAL EVAL: produced episode output -- Stages 3/4 are RUNNABLE"
else
  echo "OFFICIAL EVAL: no episode output and no crash signature -- inspect $LOG"
fi
echo "=== done: $(date) ==="
