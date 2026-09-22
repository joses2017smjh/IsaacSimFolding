#!/bin/bash
#SBATCH --job-name=lh-queue-val
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --constraint=a40|rtx8000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

set -euo pipefail
campaign=${1:?absolute campaign root}
delay=${2:?selected passing delay}
index=${SLURM_ARRAY_TASK_ID:-${3:-0}}
workspace=/nfs/hpc/share/sanchej7/Humanoid_Lite
runtime="$campaign/runtime/${SLURM_JOB_ID}_${index}"
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
  --env "SLURM_JOB_ID=$SLURM_JOB_ID" --env "SLURM_ARRAY_TASK_ID=$index" \
  --env "PYTHONPATH=$workspace/lehome51-site:$campaign/external/lehome-challenge/source/lehome:$campaign/external/lehome-challenge:$campaign/src" \
  --env "PXR_PLUGINPATH_NAME=${libs[0]}/bin/usd" "${mask[@]}" \
  "$workspace/container/bhl.sif" "$workspace/venv/bin/python" -u \
  "$campaign/scripts/run_queue_validation_task.py" --campaign "$campaign" --index "$index" --delay "$delay"
