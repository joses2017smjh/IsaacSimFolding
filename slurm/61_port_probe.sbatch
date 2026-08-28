#!/bin/bash
#SBATCH --job-name=lh-port
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|l40s|rtx8000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# How big is the port to Isaac Sim 6.0 actually?
#
# All 15 isaaclab symbols the task imports resolve on 3.0.0b2, and the particle
# cloth is raw PhysX/USD rather than isaaclab. This finds the first REAL
# failure by standing the environment up.
#
# NON-INVASIVE by construction: the extra deps go to a separate --target
# directory on PYTHONPATH, never into venv-isaac60. That venv is the BHL v60
# stack and the only thing on this cluster known to render; a resolver
# accident there would cost more than this probe is worth.

set -euo pipefail
WORKSPACE=/nfs/hpc/share/$USER/Humanoid_Lite
REPO=$WORKSPACE/lehome-fold-repro
LEHOME=$REPO/external/lehome-challenge
SIF=$WORKSPACE/container/bhl.sif
VENV=$WORKSPACE/venv-isaac60
SITE=$WORKSPACE/lehome60-site
ASSETS=$WORKSPACE/lehome-data/Assets

export XDG_CACHE_HOME=$WORKSPACE/.cache
export HOME_OVERRIDE=$WORKSPACE/.home
export UV_CACHE_DIR=$WORKSPACE/.uv-cache
mkdir -p "$SITE" "$XDG_CACHE_HOME" "$HOME_OVERRIDE"
if mkdir -p "/scratch/$USER/ov-cache" 2>/dev/null; then OV=/scratch/$USER/ov-cache
else OV=$XDG_CACHE_HOME/ov; mkdir -p "$OV"; fi

# The task resolves Assets/ relative to cwd, so make them visible in the checkout.
[ -e "$LEHOME/Assets" ] || ln -s "$ASSETS" "$LEHOME/Assets"

echo "=== node: $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

scratchbind=()
[ -d "/scratch/$USER" ] && scratchbind=(--bind "/scratch/$USER")

run() {
  apptainer exec --nv --cleanenv --home "$HOME_OVERRIDE" \
    --bind /nfs/hpc/share/$USER "${scratchbind[@]}" \
    --env OMNI_KIT_ACCEPT_EULA=YES --env ACCEPT_EULA=Y \
    --env XDG_CACHE_HOME="$XDG_CACHE_HOME" --env OV_CACHE="$OV" \
    --env UV_CACHE_DIR="$UV_CACHE_DIR" \
    --env PYTHONPATH="$SITE:$LEHOME/source:$LEHOME" \
    "$SIF" "$@"
}

if [ ! -d "$SITE/omegaconf" ]; then
    echo "--- installing light deps into $SITE (NOT into venv-isaac60) ---"
    run uv pip install --python "$VENV/bin/python" --target "$SITE" \
        omegaconf gymnasium scipy tqdm deepdiff toml
fi

set +e
run "$VENV/bin/python" "$REPO/scripts/render/port_probe.py" \
    --lehome "$LEHOME" --garment "${GARMENT:-Top_Short_Seen_0}"
rc=$?
set -e
echo "=== port probe exit: $rc ==="
case $rc in
  0) echo "RESULT: the LeHome garment env STANDS UP on Isaac Sim 6.0." ;;
  1|2) echo "RESULT: package/import level failure -- see above." ;;
  3) echo "RESULT: config built but something in the cfg moved." ;;
  4) echo "RESULT: config fine, env instantiation is where the port work is." ;;
  *) echo "RESULT: exit $rc" ;;
esac
echo "=== done: $(date) ==="
