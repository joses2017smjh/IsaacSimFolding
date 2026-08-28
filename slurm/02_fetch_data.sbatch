#!/bin/bash
#SBATCH --job-name=lh-data
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Garment assets + the released demonstrations, onto Lustre.
#
#   lehome/asset_challenge            1.0 GB   scenes, robots, garment USDs
#   lehome/dataset_challenge_merged  18.9 GB   1,000 episodes / 265,798 frames
#   lehome/dataset_challenge         24.2 GB   same, split per garment, + depth
#
# The merged set is the one the BC baseline trains on. The per-garment set is
# only needed if depth is wanted, which the paper's policy does not use.
# FETCH_DEPTH=1 pulls it anyway.
#
# These are DATA and live outside the image on purpose, so re-pulling them is
# not part of an image rebuild.

set -euo pipefail
# Data is route-independent, but the container has to exist. The train route
# reuses the already-built bhl.sif and its venv already carries huggingface_hub.
export LH_ROUTE=${LH_ROUTE:-train}
source /nfs/hpc/share/$USER/Humanoid_Lite/lehome-fold-repro/slurm/_env.sh

echo "=== node: $(hostname) | dest: $LEHOME_DATA | $(date) ==="
df -h /nfs/hpc/share/$USER | tail -1
mkdir -p "$LEHOME_DATA"
lh_exec "$REPO/slurm/inner/fetch_data.sh"
echo "=== done: $(date) ==="
