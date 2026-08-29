#!/bin/bash
#SBATCH --job-name=lh-storm51
#SBATCH --account=eecs
#SBATCH --partition=share,gpu,ampere,dgxh
#SBATCH --constraint="a40|rtx8000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Can 5.1 produce RGB without the RTX delegate that segfaults on this cluster?
#
# The pincer only holds if "5.1 cannot render" is true of 5.1 as a whole. It is
# not: what dies is librtx.scenedb.plugin.so, the RTX hydra delegate. Physics on
# 5.1 is fine -- the sibling repo ran thousands of headless jobs on it -- and
# the 5.1 wheels also ship OpenUSD's own Storm rasteriser.
#
# If Storm draws, the deviation becomes "different renderer" instead of
# "different physics", and the official geometric success checker is untouched.
# That is a far cheaper deviation than porting the cloth.
#
# Runs against the ALREADY BUILT BHL v51 venv, read-only. Deliberately does not
# import isaacsim: Kit is what loads the delegate that crashes.

set -euo pipefail
WORKSPACE=/nfs/hpc/share/$USER/Humanoid_Lite
REPO=$WORKSPACE/lehome-fold-repro
SIF=$WORKSPACE/container/bhl.sif
VENV=$WORKSPACE/venv
export XDG_CACHE_HOME=$WORKSPACE/.cache
export HOME_OVERRIDE=$WORKSPACE/.home
mkdir -p "$XDG_CACHE_HOME" "$HOME_OVERRIDE" "$REPO/results"

# pxr lives inside an extscache wheel and is not on a plain interpreter's
# sys.path; libpython has to be visible too. Same mechanism as the sibling
# repo's setup_pxr().
LIBS=$(ls -d "$VENV"/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-* 2>/dev/null | head -1)
PYLIB=$(ls -d "$WORKSPACE"/.uv-python/cpython-3.11*-linux-x86_64-gnu/lib 2>/dev/null | head -1)
[ -n "$LIBS" ] || { echo "FATAL: omni.usd.libs not found" >&2; exit 2; }
echo "=== node: $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
echo "usd libs: $LIBS"

scratchbind=()
[ -d "/scratch/$USER" ] && scratchbind=(--bind "/scratch/$USER")

set +e
apptainer exec --nv --cleanenv --home "$HOME_OVERRIDE" \
    --bind /nfs/hpc/share/$USER "${scratchbind[@]}" \
    --env PYTHONPATH="$LIBS:$REPO/src" \
    --env LD_LIBRARY_PATH="${PYLIB:+$PYLIB:}$LIBS/bin:$LIBS/lib" \
    --env PXR_PLUGINPATH_NAME="$LIBS/bin/usd" \
    --env PROBE_OUT="$REPO/results/storm51_probe.txt" \
    --env TMPDIR="${TMPDIR:-/tmp}" \
    --env EGL_PLATFORM=surfaceless \
    --env GL_PROFILE="${GL_PROFILE:-compat}" \
    --env STORM_PNG="$REPO/results/storm51_frame.png" \
    "$SIF" "$VENV/bin/python" "$REPO/scripts/render/probe_storm51.py"
rc=$?
set -e
echo "=== probe exit: $rc ==="
case $rc in
  0) echo "RESULT: Storm renders on 5.1 -- RGB without RTX is possible." ;;
  2) echo "RESULT: Storm loaded but produced nothing usable." ;;
  3) echo "RESULT: Storm path unavailable; see traceback." ;;
  139) echo "RESULT: SEGFAULT -- even the non-RTX path dies here." ;;
esac
cat "$REPO/results/storm51_probe.txt" 2>/dev/null
echo "=== done: $(date) ==="
