#!/bin/bash
#SBATCH --job-name=lh-render
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|l40s|rtx8000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Render the real challenge scene with Isaac Sim's RTX renderer -- on 6.0,
# because 5.1 cannot render on this cluster (docs/STAGE0.md).
#
# Uses the ALREADY BUILT venv-isaac60 (isaacsim 6.0.0.1, py3.12) and bhl.sif.
# Nothing here needs the LeHome pip stack: the assets are USD files and the
# renderer is Isaac Sim's own, so the 5.1 pin is irrelevant to this job. That
# is the whole reason it can run today.
#
# Constrained to cards the 6.0 probe actually rendered on. H100/H200 are
# excluded deliberately -- untested for RTX here, and a render job is not the
# place to find out.

set -euo pipefail
WORKSPACE=/nfs/hpc/share/$USER/Humanoid_Lite
REPO=$WORKSPACE/lehome-fold-repro
SIF=$WORKSPACE/container/bhl.sif
VENV=$WORKSPACE/venv-isaac60
ASSETS=$WORKSPACE/lehome-data/Assets
OUT=${OUT:-$WORKSPACE/lehome-data/render_frames}

export XDG_CACHE_HOME=$WORKSPACE/.cache
export HOME_OVERRIDE=$WORKSPACE/.home
mkdir -p "$OUT" "$XDG_CACHE_HOME" "$HOME_OVERRIDE"
if mkdir -p "/scratch/$USER/ov-cache" 2>/dev/null; then
    OV=/scratch/$USER/ov-cache
else
    OV=$XDG_CACHE_HOME/ov; mkdir -p "$OV"
fi

echo "=== node: $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

scratchbind=()
[ -d "/scratch/$USER" ] && scratchbind=(--bind "/scratch/$USER")

set +e
apptainer exec --nv --cleanenv \
    --home "$HOME_OVERRIDE" \
    --bind /nfs/hpc/share/$USER \
    "${scratchbind[@]}" \
    --env OMNI_KIT_ACCEPT_EULA=YES --env ACCEPT_EULA=Y \
    --env XDG_CACHE_HOME="$XDG_CACHE_HOME" --env OV_CACHE="$OV" \
    "$SIF" "$VENV/bin/python" "$REPO/scripts/render/render_scene.py" \
        --assets "$ASSETS" \
        --traj "$REPO/assets/demo_episode0_state.npy" \
        --out "$OUT" \
        --garment "${GARMENT:-Top_Short/Top_Short_Seen_0}" \
        --frames "${FRAMES:-120}"
rc=$?
set -e

echo "=== inner exit: $rc ==="
echo "frames written: $(ls "$OUT" 2>/dev/null | wc -l)"
ls "$OUT" 2>/dev/null | head -4
echo "=== done: $(date) ==="
