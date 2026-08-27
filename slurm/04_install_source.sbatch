#!/bin/bash
#SBATCH --job-name=lh-install
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Resolve LeHome's own pyproject into venv-lehome. This is the EDITABLE route
# (LH_ROUTE=source); prefer 01_fetch_official_image unless something in the
# environment actually has to be changed.
# Budget: isaacsim[all,extscache] is roughly 20-25 GB on disk before the forked
# Isaac Lab lands on top of it.
# A GPU is requested even though installing does not need one: the tail of this
# job imports isaacsim to prove the resolve actually produced a working stack,
# and that import touches the driver.
set -euo pipefail
export LH_ROUTE=source
source /nfs/hpc/share/$USER/Humanoid_Lite/lehome-fold-repro/slurm/_env.sh
setup_node_cache

echo "=== node: $(hostname) | target: $UV_PROJECT_ENVIRONMENT | $(date) ==="
df -h /nfs/hpc/share/$USER | tail -1
lh_exec "$REPO/slurm/inner/install_source.sh"
echo "=== done: $(date) ==="
