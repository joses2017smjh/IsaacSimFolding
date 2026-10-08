#!/bin/bash
# End-to-end only: one ordinary-H10 rollout -> persisted aligned data -> one
# mixed update -> checkpoint reload. It is not a policy-quality evaluation.
#SBATCH --job-name=lh-cl-v1-smoke
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --constraint=a40|rtx8000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail
campaign=${1:?absolute campaign root}
workspace=/nfs/hpc/share/sanchej7/Humanoid_Lite
repo="$workspace/lehome-fold-repro"
pilot="$repo/campaigns/20260921-horizon-pilot"
runtime="$campaign/runtime/${SLURM_JOB_ID}_smoke"
mkdir -p "$runtime/home" "$runtime/cache" "$runtime/ov" "$runtime/nv" "$runtime/tmp"
libs=("$workspace"/venv/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-*)
mask=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then mask+=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"); fi
export APPTAINER_CACHEDIR="$runtime/apptainer-cache"

apptainer exec --nv --cleanenv --home "$runtime/home" \
  --bind /nfs/hpc/share/sanchej7:/nfs/hpc/share/sanchej7:ro \
  --bind "$campaign:$campaign:rw" \
  --env OMNI_KIT_ACCEPT_EULA=YES --env ACCEPT_EULA=Y \
  --env "XDG_CACHE_HOME=$runtime/cache" --env "OV_CACHE=$runtime/ov" \
  --env "CUDA_CACHE_PATH=$runtime/nv" --env "TMPDIR=$runtime/tmp" \
  --env "HF_HOME=$workspace/.cache/huggingface" \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 \
  --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}" \
  --env "SLURM_JOB_ID=$SLURM_JOB_ID" \
  --env "PYTHONPATH=$workspace/lehome51-site:$pilot/external/lehome-challenge/source/lehome:$pilot/external/lehome-challenge:$repo/src" \
  --env "PXR_PLUGINPATH_NAME=${libs[0]}/bin/usd" "${mask[@]}" \
  "$workspace/container/bhl.sif" "$workspace/venv/bin/python" -u \
  "$campaign/scripts/run_rollout_task.py" --campaign "$campaign" --phase smoke --index 0 --label smoke-baseline

apptainer exec --nv --cleanenv --home "$runtime/home" \
  --bind /nfs/hpc/share/sanchej7:/nfs/hpc/share/sanchej7:ro \
  --bind "$campaign:$campaign:rw" \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 \
  --env "PYTHONPATH=$repo/src" "${mask[@]}" \
  "$workspace/container/bhl.sif" "$workspace/venv/bin/python" -u \
  "$campaign/scripts/build_rollout_dataset.py" \
  --trajectory-glob "$campaign/smoke/rollouts/*/trajectory.npz" \
  --out "$campaign/smoke/dataset.npz" \
  --provenance "$campaign/smoke/dataset-provenance.json" \
  --csv "$campaign/smoke/dataset-episodes.csv"

apptainer exec --nv --cleanenv --home "$runtime/home" \
  --bind /nfs/hpc/share/sanchej7:/nfs/hpc/share/sanchej7:ro \
  --bind "$campaign:$campaign:rw" \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 \
  --env "PYTHONPATH=$repo/src" "${mask[@]}" \
  "$workspace/container/bhl.sif" "$workspace/venv/bin/python" -u \
  "$campaign/scripts/rollout_weighted_finetune.py" \
  --policy-path /nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-data/outputs/train/bc_smolvla_raster_ft_full \
  --rollout-dataset "$campaign/smoke/dataset.npz" \
  --anchor-glob /nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-data/storm_capture/ep\*.npz \
  --anchor-files 2 --anchor-frames 2 \
  --heldout-glob /nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-data/storm_capture100/ep\*.npz \
  --heldout-files 2 --heldout-frames 2 \
  --out "$campaign/smoke/training" --steps 1 --batch-size 2 --rollout-fraction 0.5 \
  --checkpoint-every 1 --unfreeze boundary --seed 4242

apptainer exec --nv --cleanenv --home "$runtime/home" \
  --bind /nfs/hpc/share/sanchej7:/nfs/hpc/share/sanchej7:ro \
  --bind "$campaign:$campaign:rw" \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 \
  --env "PYTHONPATH=$repo/src" "${mask[@]}" \
  "$workspace/container/bhl.sif" "$workspace/venv/bin/python" -u \
  "$campaign/scripts/check_checkpoint.py" \
  --checkpoint "$campaign/smoke/training/checkpoints/step_000001" \
  --dataset "$campaign/smoke/dataset.npz" --out "$campaign/smoke/checkpoint-load.json"
