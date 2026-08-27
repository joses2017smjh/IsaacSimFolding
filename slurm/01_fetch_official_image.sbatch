#!/bin/bash
#SBATCH --job-name=lh-image
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Pull the organizers' Docker image and convert it to apptainer.
#
# This is the route that makes "success rate" mean the leaderboard's number
# rather than one this repo resolved for itself. The image is public and
# ungated on the Hub and carries the forked IsaacLab, the pinned isaacsim
# 5.1.0 and the numpy/packaging overrides already resolved:
#
#   huggingface.co/datasets/lehome/docker/lehome-challenge.tar.gz   26.7 GB
#
# Disk: ~27 GB for the tarball plus ~25 GB for the .sif, so budget ~52 GB peak
# and ~25 GB steady once the tarball is deleted. /nfs/hpc/share had 448 GB free
# and no per-user block quota when this was written -- the constraint is the
# shared filesystem, not an allocation.
#
# No GPU: this only downloads and repacks.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/lehome-fold-repro/slurm/_env.sh

TARBALL=$WORKSPACE/container/lehome-challenge.tar.gz
URL=https://huggingface.co/datasets/lehome/docker/resolve/main/lehome-challenge.tar.gz
export APPTAINER_CACHEDIR=$XDG_CACHE_HOME/apptainer
export APPTAINER_TMPDIR=$WORKSPACE/.cache/apptainer-tmp
mkdir -p "$WORKSPACE/container" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

echo "=== node: $(hostname) | $(date) ==="
df -h /nfs/hpc/share/$USER | tail -1

if [ ! -s "$TARBALL" ]; then
    echo "--- downloading 26.7 GB (resumable) ---"
    # -C - resumes a partial file: a 27 GB pull that dies at 90% should not
    # start over, and this job has been given a 6h wall clock on that basis.
    curl -L -C - --retry 5 --retry-delay 30 -o "$TARBALL" "$URL"
else
    echo "--- tarball present: $(du -h "$TARBALL" | cut -f1) ---"
fi

echo "--- converting docker-archive -> sif ---"
apptainer build --force "$SIF" "docker-archive://$TARBALL"
echo "built: $SIF ($(du -h "$SIF" | cut -f1))"

echo "--- verifying the baked venv ---"
apptainer exec "$SIF" /opt/lehome-challenge/.venv/bin/python -c "
import importlib.metadata as m
for p in ('isaacsim','isaaclab','lerobot','torch'):
    try: print(f'  {p:10s}', m.version(p))
    except Exception: print(f'  {p:10s} (absent)')
"

echo
echo "Tarball kept at $TARBALL. Delete it to reclaim 27 GB once the .sif is proven:"
echo "  rm $TARBALL"
echo "=== done: $(date) ==="
