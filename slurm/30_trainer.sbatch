#!/bin/bash
#SBATCH --job-name=lh-trainer
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Stage 3: the persistent trainer. Pair with 31_rollout_workers.
#
# Start this FIRST -- it publishes v0 into the shared directory, and the
# workers poll for that manifest rather than guessing a checkpoint path.
#
# G3 is enforced here: every rollout is checked against the manifest version
# that produced it, anything past --max_lag is dropped and counted, and the lag
# histogram is printed each cycle. Watch that histogram. A lag distribution
# that drifts upward is the silent-degradation failure arriving.
#
# --dry_run 1 exercises the whole provenance/advantage/conditioning path with
# no gradient step, which is how G3 was verified without a simulator.
set -euo pipefail
export LH_ROUTE=${LH_ROUTE:-train}
source /nfs/hpc/share/$USER/Humanoid_Lite/lehome-fold-repro/slurm/_env.sh
setup_node_cache
: "${POLICY_PATH:?set POLICY_PATH to the Stage 2 checkpoint}"
: "${VALUE_PATH:?set VALUE_PATH to the value-head checkpoint}"
export SHARED_DIR=${SHARED_DIR:-$LEHOME_DATA/stage3/shared}
export ROLLOUT_DIR=${ROLLOUT_DIR:-$LEHOME_DATA/stage3/rollouts}
export STAGE_OUT=${STAGE_OUT:-$LEHOME_DATA/stage3/checkpoints}
mkdir -p "$SHARED_DIR" "$ROLLOUT_DIR" "$STAGE_OUT"
echo "=== node: $(hostname) | shared=$SHARED_DIR | $(date) ==="
lh_exec "$REPO/slurm/inner/trainer.sh"
echo "=== done: $(date) ==="
