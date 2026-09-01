#!/bin/bash
#SBATCH --job-name=lh-bcres
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=05:30:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Resume Stage 1 BC from the 15K checkpoint toward the full 30K schedule.
#
# The original run was killed by a 6 h wall clock at step 15K -- HALF the
# planned 30,000 steps, per the checkpoint's own train_config.json. Every
# rollout so far has used that half-trained checkpoint.
#
# This is the experiment that tests the one surviving explanation for the
# policy's failure. Two earlier hypotheses died: the rasterised-vs-path-traced
# domain gap, and the wrist cameras rendering empty table. Fixing the cameras
# so all three views were correct changed the policy's behaviour almost not at
# all -- ee_z 0.640/0.635 against 0.663/0.678, cloth displacement 0.0168 m
# against 0.0144 m. A policy that indifferent to its own vision looks like one
# that learned a mean trajectory prior rather than visual servoing, which is
# what half a schedule buys.
#
# Falsifiable either way: if convergence is the issue, a policy trained to 30K
# should start responding to what it sees. If it still hovers 12 cm above the
# cloth with the same numbers, the mean-trajectory story is wrong too.
#
# 5.5 h rather than 6: MaxGRESRunMinsPerUser is already saturated by the
# user's own jobs (marl-train is pending on exactly that limit), and the cap
# counts reserved minutes, so asking for less holds less of it.

set -euo pipefail
export LH_ROUTE=${LH_ROUTE:-train}
source /nfs/hpc/share/$USER/Humanoid_Lite/lehome-fold-repro/slurm/_env.sh
setup_node_cache
export SEED=${SEED:-0}
echo "=== node: $(hostname) | resume seed=$SEED | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
lh_exec "$REPO/slurm/inner/resume_bc.sh"
echo "=== done: $(date) ==="
