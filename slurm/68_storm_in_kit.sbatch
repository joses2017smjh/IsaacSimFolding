#!/bin/bash
#SBATCH --job-name=lh-sik
#SBATCH --account=eecs
#SBATCH --partition=share,gpu,ampere,dgxh
#SBATCH --constraint="a40|rtx8000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Can Storm render inside a live Kit process? This decides whether a trained
# policy can be evaluated here at all.
#
# The env feeds the policy three RGB images. On 5.1 the RTX cameras segfault,
# so cloth_sim51 stubs them with ZEROS -- fine for measuring particles, useless
# for a VLA. Storm renders on 5.1, but so far only in a process with no Kit in
# it. If the two can share a process, the observation pipeline works and
# evaluation keeps official physics AND the official scorer.
#
# 30 minutes: this is a yes/no question.

set -euo pipefail
WORKSPACE=/nfs/hpc/share/$USER/Humanoid_Lite
REPO=$WORKSPACE/lehome-fold-repro
LEHOME=$REPO/external/lehome-challenge
SIF=$WORKSPACE/container/bhl.sif
VENV=$WORKSPACE/venv
SITE=$WORKSPACE/lehome51-site
export XDG_CACHE_HOME=$WORKSPACE/.cache HOME_OVERRIDE=$WORKSPACE/.home
mkdir -p "$XDG_CACHE_HOME" "$HOME_OVERRIDE" "$REPO/results"
if mkdir -p "/scratch/$USER/ov-cache" 2>/dev/null; then OV=/scratch/$USER/ov-cache
else OV=$XDG_CACHE_HOME/ov; mkdir -p "$OV"; fi
LIBS=$(ls -d "$VENV"/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-* | head -1)

echo "=== node: $(hostname) | $(date) ==="
scratchbind=(); [ -d "/scratch/$USER" ] && scratchbind=(--bind "/scratch/$USER")

set +e
apptainer exec --nv --cleanenv --home "$HOME_OVERRIDE" \
  --bind /nfs/hpc/share/$USER "${scratchbind[@]}" \
  --env OMNI_KIT_ACCEPT_EULA=YES --env ACCEPT_EULA=Y \
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" --env OV_CACHE="$OV" \
  --env PYTHONPATH="$SITE:$LEHOME/source/lehome:$LEHOME" \
  --env PXR_PLUGINPATH_NAME="$LIBS/bin/usd" \
  --env PROBE_OUT="$REPO/results/storm_in_kit.txt" \
  --env TMPDIR="${TMPDIR:-/tmp}" \
  "$SIF" "$VENV/bin/python" "$REPO/scripts/render/probe_storm_in_kit.py" 2>&1 | tee /tmp/sik.log
set -e

RC=$(grep -oE "storm_in_kit=(OK|DEGENERATE|EXCEPTION)" /tmp/sik.log | tail -1)
echo "=== verdict: ${RC:-no verdict printed} ==="
case "$RC" in
  *OK) echo "RESULT: in-process Storm works -> the observation pipeline is feasible." ;;
  *) echo "RESULT: not feasible in-process; Storm would have to run out-of-process." ;;
esac
echo "=== done: $(date) ==="
