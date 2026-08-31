#!/bin/bash
#SBATCH --job-name=lh-sweep
#SBATCH --account=eecs
#SBATCH --partition=share,gpu,ampere,dgxh
#SBATCH --constraint="a40|rtx8000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --array=0-7%4
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Roll the BC policy over two garments from each of the four classes and let
# LeHome's own checker label every episode. The GIF filename is written from
# that verdict, so the success/failure split cannot drift from what the
# challenge's scorer actually said.
#
# 300 steps is the horizon: the mean demonstration in the training set is
# 265.8 steps, so the policy gets slightly longer than a typical demo. A
# failure here is a real failure, not a truncated episode.
#
# Carry this with any number from this path: the policy sees RASTERISED Storm
# frames and was trained on PATH-TRACED ones.

set -euo pipefail
GARMENTS=(Top_Short_Seen_0 Top_Short_Seen_1 Top_Long_Seen_0 Top_Long_Seen_1
          Pant_Short_Seen_0 Pant_Short_Seen_1 Pant_Long_Seen_0 Pant_Long_Seen_1)
export GARMENT=${GARMENTS[$SLURM_ARRAY_TASK_ID]}
export STEPS=300
# Absolute path, NOT $(dirname "$0"): Slurm runs a COPY of this script out of
# /var/spool/slurmd/job<id>/, so $0 does not point into the repo and every
# array task died in 3 s with exit 127.
echo "=== array task $SLURM_ARRAY_TASK_ID -> $GARMENT ==="
exec bash /nfs/hpc/share/$USER/Humanoid_Lite/lehome-fold-repro/slurm/70_policy_rollout.sbatch
