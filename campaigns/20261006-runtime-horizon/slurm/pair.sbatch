#!/bin/bash
#SBATCH --job-name=lh-runtime-horizon
#SBATCH --account=eecs
#SBATCH --partition=ampere
#SBATCH --constraint=a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:55:00
set -euo pipefail
campaign=${1:?campaign root}
manifest_sha=${2:?manifest SHA256}
phase=${3:?smoke or evaluation}
workspace=/nfs/hpc/share/sanchej7/Humanoid_Lite
repo="$workspace/lehome-fold-repro"
pilot="$repo/campaigns/20260921-horizon-pilot"
runtime="$campaign/runtime/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-smoke}"
mkdir -p "$runtime/home" "$runtime/cache" "$runtime/ov" "$runtime/nv" "$runtime/tmp" "$runtime/lehome-logs"
export APPTAINER_CACHEDIR="$runtime/apptainer-cache"
libs=("$workspace"/venv/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-*)
mask=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then mask+=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"); fi
args=(--campaign "$campaign" --manifest-sha256 "$manifest_sha")
if [ "$phase" = smoke ]; then args+=(--smoke); else args+=(--index "${SLURM_ARRAY_TASK_ID:?array index}"); fi
apptainer exec --nv --cleanenv --home "$runtime/home" \
  --bind /nfs/hpc/share/sanchej7:/nfs/hpc/share/sanchej7:ro \
  --bind "$campaign:$campaign:rw" \
  --bind "$runtime/lehome-logs:$pilot/external/lehome-challenge/logs:rw" \
  --env OMNI_KIT_ACCEPT_EULA=YES --env ACCEPT_EULA=Y \
  --env "XDG_CACHE_HOME=$runtime/cache" --env "OV_CACHE=$runtime/ov" \
  --env "CUDA_CACHE_PATH=$runtime/nv" --env "TMPDIR=$runtime/tmp" \
  --env "HF_HOME=$workspace/.cache/huggingface" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 \
  --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}" \
  --env "SLURM_JOB_ID=$SLURM_JOB_ID" --env "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}" \
  --env "PYTHONPATH=$workspace/lehome51-site:$pilot/external/lehome-challenge/source/lehome:$pilot/external/lehome-challenge:$pilot/src:$repo/src" \
  --env "PXR_PLUGINPATH_NAME=${libs[0]}/bin/usd" "${mask[@]}" \
  "$workspace/container/bhl.sif" "$workspace/venv/bin/python" -u "$campaign/scripts/run_pair.py" "${args[@]}"
