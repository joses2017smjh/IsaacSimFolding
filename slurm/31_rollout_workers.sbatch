#!/bin/bash
#SBATCH --job-name=lh-worker
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --array=0-7
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Stage 3: N rollout workers, one array task each.
#
# The array IS the parallelism. LeHome's evaluator is hardcoded to a batch of
# one and its physics is CPU-only, so there is no NUM_ENVS to turn up -- see
# docs/STAGE0.md 2.3. Each task holds one environment and its own GPU slice for
# rendering and the policy forward.
#
# Each worker pins the evaluator to a Seen-only garment list before it starts.
# Collecting on the two public Unseen garments per type and training on the
# result is the leak that makes the generalisation number meaningless, and
# nothing downstream would notice.
#
# BLOCKED until a route can render -- see results/rtx51_probe_borrowed.txt.
set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/lehome-fold-repro/slurm/_env.sh
setup_node_cache
export SHARED_DIR=${SHARED_DIR:-$LEHOME_DATA/stage3/shared}
export ROLLOUT_DIR=${ROLLOUT_DIR:-$LEHOME_DATA/stage3/rollouts}
export WORKER_ID=${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$SHARED_DIR" "$ROLLOUT_DIR"
echo "=== node: $(hostname) | worker $WORKER_ID | $(date) ==="
lh_exec "$REPO/slurm/inner/rollout_worker.sh"
echo "=== done: $(date) ==="
