#!/bin/bash
#SBATCH --job-name=lh-build
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Build the runtime image. No GPU needed -- the driver comes from the host at
# run time via --nv, and nothing in %post touches CUDA.
set -euo pipefail
export LH_ROUTE=source
source /nfs/hpc/share/$USER/Humanoid_Lite/lehome-fold-repro/slurm/_env.sh

export APPTAINER_CACHEDIR=$XDG_CACHE_HOME/apptainer
export APPTAINER_TMPDIR=${TMPDIR:-/tmp}/apptainer-$USER
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$WORKSPACE/container"

echo "=== node: $(hostname) | $(date) ==="
apptainer build --force "$SIF" "$REPO/container/lehome.def"
echo "built: $SIF  ($(du -h "$SIF" | cut -f1))"
echo "=== done: $(date) ==="
