#!/bin/bash
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --job-name=lh-eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Evaluate any stage's checkpoint through the OFFICIAL harness.
#   POLICY_TYPE=lerobot    Stage 1 BC
#   POLICY_TYPE=candidate  Stage 2, value-head candidate selection
#   POLICY_TYPE=recap      Stage 3, advantage-conditioned
#
# BLOCKED until a route can render -- see results/rtx51_probe_borrowed.txt.
# Training does not need this job; measuring does.
set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/lehome-fold-repro/slurm/_env.sh
setup_node_cache
export POLICY_TYPE=${POLICY_TYPE:-lerobot}
export GARMENT_TYPE=${GARMENT_TYPE:-top_long}
export NUM_EPISODES=${NUM_EPISODES:-5}
export MAX_STEPS=${MAX_STEPS:-600}
export SEED=${SEED:-0}
echo "=== node: $(hostname) | $POLICY_TYPE / $GARMENT_TYPE seed=$SEED | $(date) ==="
lh_exec "$REPO/slurm/inner/eval_stage.sh"
echo "=== done: $(date) ==="
