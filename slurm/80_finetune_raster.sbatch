#!/bin/bash
#SBATCH --job-name=lh-ft
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Fine-tune on RASTERISED frames -- the last untested lever.
#
# Measured, twice, at two checkpoints:
#   15K  path-traced +0.966  Storm +0.063  gap 0.902
#   30K  path-traced +0.976  Storm -0.038  gap 1.014
# Training helped in distribution and hurt out of it. The renderer is what is
# left, and 3,200 (Storm frame, demonstration action) pairs are now on disk.
#
# Starts from the CONVERGED 30K checkpoint: it is the better policy on the
# task, and the question is whether its representation can be moved onto the
# renderer it will be deployed on.

set -euo pipefail
export LH_ROUTE=${LH_ROUTE:-train}
source /nfs/hpc/share/$USER/Humanoid_Lite/lehome-fold-repro/slurm/_env.sh
setup_node_cache
export BASE_CKPT=${BASE_CKPT:-$LEHOME_DATA/outputs/train/bc_smolvla_seed0/checkpoints/030000/pretrained_model}
export CAP_GLOB=${CAP_GLOB:-$LEHOME_DATA/storm_capture/*.npz}
export FT_OUT=${FT_OUT:-$LEHOME_DATA/outputs/train/bc_smolvla_raster_ft}
export FT_STEPS=${FT_STEPS:-1500} FT_UNFREEZE=${FT_UNFREEZE:-vision}
echo "=== node: $(hostname) | $(date) ==="
echo "    base: $BASE_CKPT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
lh_exec "$REPO/slurm/inner/finetune_raster.sh"
echo "=== done: $(date) ==="
