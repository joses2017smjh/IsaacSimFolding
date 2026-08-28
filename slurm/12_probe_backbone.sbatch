#!/bin/bash
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --job-name=lh-probe-bb
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Find, once, where the pooled backbone feature lives -- then PIN it.
# The value heads read that feature; a training loop that searches for it at
# runtime discovers the mistake hours in. Run this after Stage 1, read the
# output, put the answer in FEATURE_PATH / HIDDEN_DIM.
set -euo pipefail
export LH_ROUTE=${LH_ROUTE:-train}
source /nfs/hpc/share/$USER/Humanoid_Lite/lehome-fold-repro/slurm/_env.sh
setup_node_cache
: "${POLICY_PATH:?set POLICY_PATH to a Stage 1 checkpoint}"
echo "=== node: $(hostname) | $POLICY_PATH | $(date) ==="
lh_exec "$REPO/slurm/inner/probe_backbone.sh"
