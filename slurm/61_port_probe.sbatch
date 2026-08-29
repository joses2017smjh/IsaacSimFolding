#!/bin/bash
#SBATCH --job-name=lh-port
#SBATCH --account=eecs
#SBATCH --partition=share,gpu,ampere,dgxh
#SBATCH --constraint="a40|rtx8000"
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
# The dependency list grows by iteration on purpose. lehome/devices/__init__
# imports the SO101 *leader* (teleop hardware) at package import time, so the
# task module drags in pynput, pyserial and friends even though nothing here
# teleoperates. Each run names the next missing one; that is cheaper than
# guessing the closure up front, and it documents exactly what the task needs.
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

# The task resolves Assets/ relative to cwd, so the data has to appear inside
# the checkout. Link the CONTENTS, not the directory: upstream ships
# Assets/.gitignore as a placeholder, so the directory already exists and is
# not empty. `rmdir` fails on it, and `ln -s src Assets` quietly creates
# Assets/Assets instead of replacing anything -- which is how the loader came
# to report "Garment directory not found" for a garment that was on disk.
mkdir -p "$LEHOME/Assets"
for d in "$ASSETS"/*; do
    n=$(basename "$d")
    [ "$n" = ".gitignore" ] && continue
    if [ -L "$LEHOME/Assets/$n" ]; then
        [ "$(readlink -f "$LEHOME/Assets/$n")" = "$(readlink -f "$d")" ] || \
            { rm -f "$LEHOME/Assets/$n"; ln -s "$d" "$LEHOME/Assets/$n"; }
    elif [ ! -e "$LEHOME/Assets/$n" ]; then
        ln -s "$d" "$LEHOME/Assets/$n"
    fi
done
ls -d "$LEHOME/Assets/objects/Challenge_Garment/Release" >/dev/null 2>&1 \
    && echo "Assets visible at \$LEHOME/Assets" \
    || { echo "FATAL: Assets not reachable from the checkout" >&2; exit 2; }

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
    --env PYTHONPATH="$SITE:$LEHOME/source/lehome:$LEHOME" \
    "$SIF" "$@"
}

if [ ! -d "$SITE/pynput" ]; then
    echo "--- installing light deps into $SITE (NOT into venv-isaac60) ---"
    run uv pip install --python "$VENV/bin/python" --target "$SITE" \
        omegaconf gymnasium scipy tqdm deepdiff toml plotly open3d \
        pynput pyserial num2words transforms3d
fi

# Kit swallows sys.exit(): the previous run had the probe print
# "PORT PROBE EXIT 3" while apptainer returned 0, and this script cheerfully
# announced that the environment stands up. It does not. Read the verdict the
# probe PRINTS, not the status the app returns.
LOG=$WORKSPACE/logs/lh-port-${SLURM_JOB_ID}.probe
set +e
run "$VENV/bin/python" "$REPO/scripts/render/port_probe.py" \
    --lehome "$LEHOME" --garment "${GARMENT:-Top_Short_Seen_0}" 2>&1 | tee "$LOG"
set -e
rc=$(grep -oE "PORT PROBE EXIT [0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+$")
if [ -z "$rc" ]; then
    echo "RESULT: the probe never printed a verdict -- it died before step 1."
    exit 1
fi
echo "=== port probe exit: $rc (read from the probe's own verdict line) ==="
case $rc in
  0) echo "RESULT: the LeHome garment env STANDS UP on Isaac Sim 6.0." ;;
  1|2) echo "RESULT: package/import level failure -- see above." ;;
  3) echo "RESULT: config built but something in the cfg moved." ;;
  4) echo "RESULT: config fine, env instantiation is where the port work is." ;;
  *) echo "RESULT: exit $rc" ;;
esac
echo "=== done: $(date) ==="
