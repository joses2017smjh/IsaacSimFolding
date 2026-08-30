#!/bin/bash
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --job-name=lh-bc
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

# Stage 1: BC fine-tune on the 10 Seen garments per type.
#
# The released demonstrations contain ONLY Seen garments (40 of them, 25
# episodes each), so this cannot leak into the validation split even by
# accident -- verified against meta/garment_info.json. The leak exposure is
# Stage 3, where rollouts are collected; rollout_worker.py guards that.
#
# 6 hours, not 24. The account is capped on GPU-MINUTES
# (MaxGRESRunMinsPerAccount), not just concurrent GPUs, so a 24h request
# reserves 1440 GPU-min and blocks both this job and everything else on the
# account -- including the user's own v2-train array. save_freq=5000 writes
# intermediate checkpoints, so a shorter wall clock still yields a usable
# policy; it just yields fewer steps of one.
#
# CONFIG picks the base model, and the choice is not cosmetic:
#   configs/bc_smolvla.yaml   ~450M, cheap, NOT the paper's model
#   configs/bc_pi05.yaml      ~2.3B, the paper's model
# The run name carries it so no number can be misattributed later.
set -euo pipefail
export LH_ROUTE=${LH_ROUTE:-train}
source /nfs/hpc/share/$USER/Humanoid_Lite/lehome-fold-repro/slurm/_env.sh
setup_node_cache
export CONFIG=${CONFIG:-configs/bc_smolvla.yaml}
export SEED=${SEED:-0}
echo "=== node: $(hostname) | $CONFIG seed=$SEED | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
lh_exec "$REPO/slurm/inner/train_bc.sh"
echo "=== done: $(date) ==="
