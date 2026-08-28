#!/bin/bash
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --job-name=lh-trainenv
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

# lerobot + torch, no isaacsim. ~8 GB against the full stack's ~25 GB.
#
# Worth its own job because of what 00_rtx_probe found: Isaac Sim 5.1 cannot
# render here, which blocks evaluation -- and blocks nothing about Stage 1 or
# Stage 2 TRAINING, which only ever touch the released LeRobot dataset.
set -euo pipefail
export LH_ROUTE=train
source /nfs/hpc/share/$USER/Humanoid_Lite/lehome-fold-repro/slurm/_env.sh
setup_node_cache
echo "=== node: $(hostname) | target: $UV_PROJECT_ENVIRONMENT | $(date) ==="
lh_exec "$REPO/slurm/inner/install_train_venv.sh"
echo "=== done: $(date) ==="
