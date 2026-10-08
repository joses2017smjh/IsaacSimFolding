#!/bin/bash
# One fixed-seed, 300-step, 50/50 AWR-resampled rollout + BC-anchor update.
#SBATCH --job-name=lh-cl-v1-train
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --constraint=a40|rtx8000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

set -euo pipefail
campaign=${1:?absolute campaign root}
workspace=/nfs/hpc/share/sanchej7/Humanoid_Lite
repo="$workspace/lehome-fold-repro"
runtime="$campaign/runtime/${SLURM_JOB_ID}_train"
mkdir -p "$runtime/home" "$runtime/cache" "$runtime/ov" "$runtime/nv" "$runtime/tmp"
mask=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then mask+=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"); fi
export APPTAINER_CACHEDIR="$runtime/apptainer-cache"
apptainer exec --nv --cleanenv --home "$runtime/home" \
  --bind /nfs/hpc/share/sanchej7:/nfs/hpc/share/sanchej7:ro \
  --bind "$campaign:$campaign:rw" \
  --env "XDG_CACHE_HOME=$runtime/cache" --env "CUDA_CACHE_PATH=$runtime/nv" --env "TMPDIR=$runtime/tmp" \
  --env "HF_HOME=$workspace/.cache/huggingface" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 \
  --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}" --env "SLURM_JOB_ID=$SLURM_JOB_ID" \
  --env "PYTHONPATH=$repo/src" "${mask[@]}" \
  "$workspace/container/bhl.sif" "$workspace/venv/bin/python" -u \
  "$campaign/scripts/rollout_weighted_finetune.py" \
  --policy-path /nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-data/outputs/train/bc_smolvla_raster_ft_full \
  --rollout-dataset "$campaign/datasets/iteration1.npz" \
  --anchor-glob /nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-data/storm_capture/ep\*.npz \
  --anchor-files 16 --anchor-frames 8 \
  --heldout-glob /nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-data/storm_capture100/ep\*.npz \
  --heldout-files 4 --heldout-frames 8 \
  --out "$campaign/training/iteration1" --steps 300 --batch-size 4 --rollout-fraction 0.5 \
  --checkpoint-every 100 --unfreeze boundary --seed 4242
