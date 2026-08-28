#!/bin/bash
#SBATCH --job-name=lh-tune
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Stage 4: Thompson sampling over inference hyperparameters. No training.
# Reports gain over fixed defaults on the SAME checkpoint, with the default arm
# given a reserved budget so the baseline is measured rather than starved.
set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/lehome-fold-repro/slurm/_env.sh
setup_node_cache
: "${POLICY_PATH:?set POLICY_PATH}"
: "${VALUE_PATH:?set VALUE_PATH}"
echo "=== node: $(hostname) | budget ${BUDGET:-400} episodes | $(date) ==="
lh_exec "$REPO/slurm/inner/thompson.sh"
echo "=== done: $(date) ==="
