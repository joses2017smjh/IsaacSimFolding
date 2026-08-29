#!/bin/bash
#SBATCH --job-name=lh-gallery
#SBATCH --account=eecs
#SBATCH --partition=share,gpu,ampere,dgxh
#SBATCH --constraint="a40|rtx8000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Render the same scene once per known defect, plus once corrected, from an
# identical camera -- so each fix can be SHOWN against the bug it fixed rather
# than asserted in a commit message.
#
# Every defect here is one this renderer actually shipped and I actually had to
# find. They are reintroduced by a flag, not re-broken by hand, so the gallery
# regenerates from source and cannot drift from the claims made about it.

set -euo pipefail
WORKSPACE=/nfs/hpc/share/$USER/Humanoid_Lite
REPO=$WORKSPACE/lehome-fold-repro
SIF=$WORKSPACE/container/bhl.sif
VENV=$WORKSPACE/venv-isaac60
ASSETS=$WORKSPACE/lehome-data/Assets
OUT=${OUT:-$REPO/docs/img/gallery}

export XDG_CACHE_HOME=$WORKSPACE/.cache
export HOME_OVERRIDE=$WORKSPACE/.home
mkdir -p "$OUT" "$XDG_CACHE_HOME" "$HOME_OVERRIDE"
if mkdir -p "/scratch/$USER/ov-cache" 2>/dev/null; then OV=/scratch/$USER/ov-cache
else OV=$XDG_CACHE_HOME/ov; mkdir -p "$OV"; fi

echo "=== node: $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

scratchbind=()
[ -d "/scratch/$USER" ] && scratchbind=(--bind "/scratch/$USER")

render_one() {
    local defect=$1
    echo; echo "--- rendering defect=$defect ---"
    set +e
    apptainer exec --nv --cleanenv --home "$HOME_OVERRIDE" \
        --bind /nfs/hpc/share/$USER "${scratchbind[@]}" \
        --env OMNI_KIT_ACCEPT_EULA=YES --env ACCEPT_EULA=Y \
        --env XDG_CACHE_HOME="$XDG_CACHE_HOME" --env OV_CACHE="$OV" \
        "$SIF" "$VENV/bin/python" "$REPO/scripts/render/render_scene.py" \
            --assets "$ASSETS" \
            --traj "$REPO/assets/demo_episode0_state.npy" \
            --out "$OUT" --defect "$defect" --stills 1 \
            --width 960 --height 540 2>&1 | grep -E "^\[render\]|Error|Traceback"
    local rc=${PIPESTATUS[0]}
    set -e
    echo "--- defect=$defect exit $rc ---"
}

for d in none exposure framing gravity garment_scale garment_culling no_yaw; do
    render_one "$d"
done

echo; echo "=== gallery ==="
ls -la "$OUT"
echo "=== done: $(date) ==="
