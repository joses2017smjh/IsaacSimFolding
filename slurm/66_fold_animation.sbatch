#!/bin/bash
#SBATCH --job-name=lh-fold
#SBATCH --account=eecs
#SBATCH --partition=share,gpu,ampere,dgxh
#SBATCH --constraint="a40|rtx8000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Render the SIMULATED cloth as an animation.
#
# results/cloth51.npz is real PhysX particle cloth from 5.1 -- (365, 9774, 3)
# world-space particle positions driven by all 364 steps of episode 0 of the
# released demonstrations. Each frame writes that frame's positions into the
# garment mesh and rasterises through Storm, which 5.1 can do without the RTX
# delegate that segfaults here.
#
# The geometry is simulation output, not an authored pose. That is the whole
# difference from every earlier render in this repo.

set -euo pipefail
WORKSPACE=/nfs/hpc/share/$USER/Humanoid_Lite
REPO=$WORKSPACE/lehome-fold-repro
SIF=$WORKSPACE/container/bhl.sif
VENV=$WORKSPACE/venv
ASSETS=$WORKSPACE/lehome-data/Assets
OUT=${OUT:-$WORKSPACE/lehome-data/fold_frames}

export XDG_CACHE_HOME=$WORKSPACE/.cache
export HOME_OVERRIDE=$WORKSPACE/.home
mkdir -p "$OUT" "$XDG_CACHE_HOME" "$HOME_OVERRIDE"
rm -f "$OUT"/*.png

LIBS=$(ls -d "$VENV"/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-* | head -1)
PYLIB=$(ls -d "$WORKSPACE"/.uv-python/cpython-3.11*-linux-x86_64-gnu/lib 2>/dev/null | head -1)

echo "=== node: $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

scratchbind=()
[ -d "/scratch/$USER" ] && scratchbind=(--bind "/scratch/$USER")

set +e
apptainer exec --nv --cleanenv --home "$HOME_OVERRIDE" \
    --bind /nfs/hpc/share/$USER "${scratchbind[@]}" \
    --env PYTHONPATH="$LIBS" \
    --env LD_LIBRARY_PATH="${PYLIB:+$PYLIB:}$LIBS/bin:$LIBS/lib" \
    --env PXR_PLUGINPATH_NAME="$LIBS/bin/usd" \
    --env TMPDIR="${TMPDIR:-/tmp}" \
    "$SIF" "$VENV/bin/python" "$REPO/scripts/render/fold_animation.py" \
        --particles "$REPO/results/cloth51.npz" \
        --assets "$ASSETS" \
        --garment "${GARMENT:-Top_Short/Top_Short_Seen_0}" \
        --out "$OUT" --width "${WIDTH:-800}" --stride "${STRIDE:-3}"
rc=$?
set -e
echo "=== exit: $rc | frames: $(ls "$OUT"/*.png 2>/dev/null | wc -l) ==="
echo "=== done: $(date) ==="
