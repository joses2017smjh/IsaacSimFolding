#!/bin/bash
#SBATCH --job-name=lh-stormscene
#SBATCH --account=eecs
#SBATCH --partition=share,gpu,ampere,dgxh
#SBATCH --constraint="a40|rtx8000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# The real challenge scene, rendered on the 5.1 stack via Storm -- no Kit, no
# SimulationApp, no RTX delegate. 5.1 is the stack that HAS the particle cloth,
# so if this draws, the "cannot both simulate and render" conclusion in
# docs/STAGE0.md is wrong and the project has a path forward that keeps the
# official physics and the official scorer.

set -euo pipefail
WORKSPACE=/nfs/hpc/share/$USER/Humanoid_Lite
REPO=$WORKSPACE/lehome-fold-repro
SIF=$WORKSPACE/container/bhl.sif
VENV=$WORKSPACE/venv
ASSETS=$WORKSPACE/lehome-data/Assets
export XDG_CACHE_HOME=$WORKSPACE/.cache
export HOME_OVERRIDE=$WORKSPACE/.home
mkdir -p "$XDG_CACHE_HOME" "$HOME_OVERRIDE" "$REPO/results"

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
    "$SIF" "$VENV/bin/python" "$REPO/scripts/render/storm_scene.py" \
        --assets "$ASSETS" \
        --garment "${GARMENT:-Top_Short/Top_Short_Seen_0}" \
        --out "$REPO/docs/img/storm51_scene.png" \
        --width "${WIDTH:-960}"
rc=$?
set -e
echo "=== exit: $rc ==="
[ $rc -eq 0 ] && echo "RESULT: the real challenge scene renders on 5.1 without RTX."
echo "=== done: $(date) ==="
