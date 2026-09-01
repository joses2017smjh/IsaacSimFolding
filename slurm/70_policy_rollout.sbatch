#!/bin/bash
#SBATCH --job-name=lh-roll
#SBATCH --account=eecs
#SBATCH --partition=share,gpu,ampere,dgxh
#SBATCH --constraint="a40|rtx8000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# The first policy-driven episode in this project.
#
# Everything this needs was established separately and is assembled here:
# the LeHome env and its particle cloth run on 5.1 (sim.device must be CUDA);
# Storm rasterises without the RTX delegate that segfaults here, and does so
# inside a live Kit process; and the trained SmolVLA checkpoint loads in that
# same interpreter once transformers/tokenizers/hub are pinned to the versions
# it was TRAINED against.
#
# The verdict comes from LeHome's own _get_success -- geometric over particle
# positions -- so the label on the episode is the challenge's, not mine.
#
# Deviation, restated wherever a number from this path appears: the policy is
# shown RASTERISED frames and was trained on PATH-TRACED ones. If it scores
# badly, domain gap is a live explanation that has to be ruled out before
# blaming the policy.

set -euo pipefail
WORKSPACE=/nfs/hpc/share/$USER/Humanoid_Lite
REPO=$WORKSPACE/lehome-fold-repro
LEHOME=$REPO/external/lehome-challenge
SIF=$WORKSPACE/container/bhl.sif
VENV=$WORKSPACE/venv
SITE=$WORKSPACE/lehome51-site
ASSETS=$WORKSPACE/lehome-data/Assets
CKPT=${CKPT:-$WORKSPACE/lehome-data/outputs/train/bc_smolvla_seed0/checkpoints/015000/pretrained_model}
GARMENT=${GARMENT:-Top_Short_Seen_0}
GDIR=$ASSETS/objects/Challenge_Garment/Release/${GARMENT%_Seen_*}/$GARMENT
# Per-RUN workdir, not per-garment. Storm renders into fixed filenames
# (top_rgb.png, obs_stage.usda, ...) inside this directory, so two concurrent
# array tasks on the SAME garment -- different demo episodes of it -- raced and
# clobbered each other mid-render, surfacing as UnidentifiedImageError and a
# missing obs_stage.usda. RESULT_SUFFIX already distinguishes the episode.
FRAMES=${FRAMES:-$WORKSPACE/lehome-data/rollout_frames/${GARMENT}${RESULT_SUFFIX:-}}

export XDG_CACHE_HOME=$WORKSPACE/.cache HOME_OVERRIDE=$WORKSPACE/.home
export HF_HOME=$WORKSPACE/.cache/huggingface
mkdir -p "$XDG_CACHE_HOME" "$HOME_OVERRIDE" "$FRAMES" "$REPO/results"
if mkdir -p "/scratch/$USER/ov-cache" 2>/dev/null; then OV=/scratch/$USER/ov-cache
else OV=$XDG_CACHE_HOME/ov; mkdir -p "$OV"; fi
LIBS=$(ls -d "$VENV"/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-* | head -1)

mkdir -p "$LEHOME/Assets"
for d in "$ASSETS"/*; do n=$(basename "$d"); [ "$n" = ".gitignore" ] && continue
  [ -e "$LEHOME/Assets/$n" ] || ln -s "$d" "$LEHOME/Assets/$n"; done

echo "=== node: $(hostname) | garment: $GARMENT | $(date) ==="
echo "checkpoint: $CKPT"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

scratchbind=(); [ -d "/scratch/$USER" ] && scratchbind=(--bind "/scratch/$USER")
PLOG=$WORKSPACE/logs/lh-roll-${SLURM_JOB_ID}.probe
set +e
apptainer exec --nv --cleanenv --home "$HOME_OVERRIDE" \
  --bind /nfs/hpc/share/$USER "${scratchbind[@]}" \
  --env OMNI_KIT_ACCEPT_EULA=YES --env ACCEPT_EULA=Y \
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" --env OV_CACHE="$OV" --env HF_HOME="$HF_HOME" \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env PYTHONPATH="$SITE:$LEHOME/source/lehome:$LEHOME:$REPO/src" \
  --env PXR_PLUGINPATH_NAME="$LIBS/bin/usd" \
  --env PYTHONUNBUFFERED=1 --env TMPDIR="${TMPDIR:-/tmp}" \
  "$SIF" "$VENV/bin/python" -u "$REPO/scripts/render/policy_rollout51.py" \
    --lehome "$LEHOME" --policy_path "$CKPT" \
    --garment "$GARMENT" --garment_dir "$GDIR" --assets "$ASSETS" \
    --steps "${STEPS:-300}" --frames_out "$FRAMES" \
    --settle_steps "${SETTLE:-0}" --shadow_policy "${SHADOW:-0}" \
    ${REPLAY:+--replay_parquet "$REPLAY" --replay_episode "${REPLAY_EP:-0}"} \
    ${MATCH_POSE:+--match_pose="$MATCH_POSE"} ${MATCH_SCALE:+--match_scale="$MATCH_SCALE"} \
    --result_out "$REPO/results/rollout_${GARMENT}${RESULT_SUFFIX:-}.json" 2>&1 | tee "$PLOG"
set -e

V=$(grep -oE "VERDICT rollout success=(True|False)" "$PLOG" | tail -1)
echo "=== ${V:-no verdict printed} ==="
echo "=== done: $(date) ==="
