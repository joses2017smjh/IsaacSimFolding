#!/bin/bash
#SBATCH --job-name=lh-cloth51
#SBATCH --account=eecs
#SBATCH --partition=share,gpu,ampere,dgxh
#SBATCH --constraint="a40|rtx8000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Simulate the garment for real: PhysX particle cloth on Isaac Sim 5.1.
#
# Everything rendered so far has been the garment's AUTHORED mesh, which is why
# it looks inflated -- it has never been simulated. 5.1 is the stack that still
# has particle cloth, and its crash is in the RENDERER, so launching with
# cameras OFF avoids it entirely. That is exactly how bhl-robustness-ladder ran
# thousands of headless 5.1 jobs.
#
# Uses the ALREADY BUILT v51 venv plus a separate --target site dir for
# lehome's light dependencies, so the locked venv is never written to.

set -euo pipefail
WORKSPACE=/nfs/hpc/share/$USER/Humanoid_Lite
REPO=$WORKSPACE/lehome-fold-repro
LEHOME=$REPO/external/lehome-challenge
SIF=$WORKSPACE/container/bhl.sif
VENV=$WORKSPACE/venv
SITE=$WORKSPACE/lehome51-site
ASSETS=$WORKSPACE/lehome-data/Assets

export XDG_CACHE_HOME=$WORKSPACE/.cache
export HOME_OVERRIDE=$WORKSPACE/.home
export UV_CACHE_DIR=$WORKSPACE/.uv-cache
mkdir -p "$SITE" "$XDG_CACHE_HOME" "$HOME_OVERRIDE" "$REPO/results"
if mkdir -p "/scratch/$USER/ov-cache" 2>/dev/null; then OV=/scratch/$USER/ov-cache
else OV=$XDG_CACHE_HOME/ov; mkdir -p "$OV"; fi

# Assets must be visible from the checkout: the loader resolves them from cwd.
# Link the CONTENTS -- upstream ships Assets/.gitignore, so the directory
# exists and is not empty.
mkdir -p "$LEHOME/Assets"
for d in "$ASSETS"/*; do
    n=$(basename "$d"); [ "$n" = ".gitignore" ] && continue
    [ -e "$LEHOME/Assets/$n" ] || ln -s "$d" "$LEHOME/Assets/$n"
done

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

# open3d is in this list because get_current_mesh_points() imports it, and
# that is the accessor the OFFICIAL success checker uses -- without it the
# scorer's own read path is unavailable and the fallback hits a 5.1 API quirk
# ('ClothPrim' object has no attribute '_max_particles_per_cloth').
if [ ! -d "$SITE/open3d" ]; then
    echo "--- installing lehome's light deps into $SITE (py3.11, NOT into venv) ---"
    run uv pip install --python "$VENV/bin/python" --target "$SITE" \
        omegaconf gymnasium tqdm deepdiff toml plotly pynput pyserial \
        num2words transforms3d open3d
fi

# Kit swallows sys.exit(), so the process status is NOT the probe's verdict --
# the first run printed "the particle cloth SIMULATES" over a traceback because
# apptainer returned 0. Read the verdict the probe PRINTS. Same bug, same fix,
# as 61_port_probe.
PLOG=$WORKSPACE/logs/lh-cloth51-${SLURM_JOB_ID}.probe
set +e
run "$VENV/bin/python" "$REPO/scripts/render/cloth_sim51.py" \
    --lehome "$LEHOME" --garment "${GARMENT:-Top_Short_Seen_0}" \
    --steps "${STEPS:-120}" --out "$REPO/results/cloth51.npz" \
    --traj "$REPO/assets/demo_episode0_state.npy" 2>&1 | tee "$PLOG"
set -e
if grep -q "^\[cloth\] VERDICT cloth=SIMULATING" "$PLOG"; then rc=0
elif grep -q "^\[cloth\] VERDICT cloth=STATIC" "$PLOG"; then rc=2
elif grep -q "^\[cloth\] TRACEBACK" "$PLOG"; then rc=3
else rc=4; fi
echo "=== verdict-derived exit: $rc ==="
case $rc in
  0) echo "RESULT: the particle cloth SIMULATES on 5.1. Folding animation is now a rendering problem." ;;
  2) echo "RESULT: env stood up but the cloth did not move." ;;
  3) echo "RESULT: exception -- see traceback." ;;
  4) echo "RESULT: could not read particle positions." ;;
  139) echo "RESULT: SEGFAULT even with cameras off." ;;
esac
echo "=== done: $(date) ==="
