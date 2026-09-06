#!/bin/bash
#SBATCH --job-name=lh-evalprobe
#SBATCH --account=eecs
#SBATCH --partition=share,gpu,ampere,dgxh
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Does the OFFICIAL evaluator run on this stack?
#
# Stages 3 and 4 both route through scripts/run_eval.py into LeHome's own
# scripts.eval with --enable_cameras -- the Isaac Lab render product that 5.1's
# RTX delegate segfaults on. That is why every rollout here uses Storm instead.
#
# The first version of this probe reported "crashed (RTX path)" when apptainer
# had failed to open a container that was never built: LH_ROUTE=source wants
# lehome.sif and only bhl.sif exists. It never reached Isaac Sim. The grep
# matched the word Fatal and attributed it to RTX, which is the exact
# false-confidence failure the probe was written to avoid. So this one calls
# apptainer directly with the container the working rollouts use, and its
# verdict distinguishes "never started" from "started and crashed".

set -euo pipefail
W=/nfs/hpc/share/$USER/Humanoid_Lite
REPO=$W/lehome-fold-repro
LEHOME=$REPO/external/lehome-challenge
SIF=$W/container/bhl.sif
VENV=$W/venv
SITE=$W/lehome51-site
ASSETS=$W/lehome-data/Assets
CKPT=${POLICY_PATH:-$W/lehome-data/outputs/train/bc_smolvla_raster_ft_full}
LOG=$W/logs/evalprobe-${SLURM_JOB_ID}.txt

export XDG_CACHE_HOME=$W/.cache HF_HOME=$W/.cache/huggingface
mkdir -p "$XDG_CACHE_HOME"
LIBS=$(ls -d "$VENV"/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-* | head -1)
mkdir -p "$LEHOME/Assets"
for d in "$ASSETS"/*; do n=$(basename "$d"); [ "$n" = ".gitignore" ] && continue
  [ -e "$LEHOME/Assets/$n" ] || ln -s "$d" "$LEHOME/Assets/$n"; done
for d in Datasets outputs; do
  [ -e "$LEHOME/$d" ] || ln -s "$W/lehome-data/$d" "$LEHOME/$d" 2>/dev/null || true; done

echo "=== node: $(hostname) | $(date) ==="
echo "    container: $SIF ($([ -f "$SIF" ] && echo present || echo MISSING))"
set +e
apptainer exec --nv --cleanenv --home "$W/.home" --bind /nfs/hpc/share/$USER \
  --env OMNI_KIT_ACCEPT_EULA=YES --env ACCEPT_EULA=Y \
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" --env HF_HOME="$HF_HOME" \
  --env PYTHONPATH="$SITE:$LEHOME/source/lehome:$LEHOME:$REPO/src:$REPO/scripts" \
  --env PXR_PLUGINPATH_NAME="$LIBS/bin/usd" --env PYTHONUNBUFFERED=1 \
  "$SIF" "$VENV/bin/python" -u "$REPO/scripts/run_eval.py" \
    --policy_type lerobot --policy_path "$CKPT" \
    --dataset_root Datasets/example/four_types_merged \
    --garment_type pant_short --num_episodes 1 --max_steps 60 \
    --device cpu --enable_cameras --headless 2>&1 | tee "$LOG"
set -e

echo "--- verdict ---"
if ! [ -s "$LOG" ]; then
  echo "INCONCLUSIVE: no output at all"
elif grep -qiE "could not open image|FATAL: *While|no such file.*\.sif" "$LOG"; then
  echo "INCONCLUSIVE: container/apptainer failure -- the evaluator never started"
elif grep -qiE "Segmentation fault|librtx|core dumped" "$LOG"; then
  echo "BLOCKED: reached Isaac Sim and crashed in the RTX path"
elif grep -qiE "episode|success|Final result" "$LOG"; then
  echo "RUNNABLE: the official evaluator produced episode output"
else
  echo "INCONCLUSIVE: started but produced neither episodes nor a crash signature"
  echo "  last lines:"; tail -6 "$LOG" | sed 's/^/    /'
fi
echo "=== done: $(date) ==="
