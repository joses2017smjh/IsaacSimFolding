#!/bin/bash
# Fixed-seed mixed corrective/raster replay, capped at 300 optimizer steps.
# One corrective example plus four disjoint original-raster TRAIN frames per
# batch gives exactly 20/80; checkpoints are retained every 25 steps.
# The original baseline checkpoint is read-only and never overwritten.
#SBATCH --job-name=lh-mixed-replay-train
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --constraint=a40|rtx8000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail
campaign=${1:?absolute campaign root}
workspace=/nfs/hpc/share/sanchej7/Humanoid_Lite
source_checkpoint=/nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-data/outputs/train/bc_smolvla_raster_ft_full
boundary_capture="$campaign/analysis/boundary-replay/pose3-boundary-replay.npz"
heldout_glob=/nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-data/storm_capture100/ep*.npz
raster_replay_glob=/nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-data/storm_capture/ep*.npz
out="$campaign/analysis/mixed-replay"
runtime="$campaign/runtime/${SLURM_JOB_ID}_mixed_replay_train"
mkdir -p "$runtime/home" "$runtime/cache" "$runtime/ov" "$runtime/nv" "$runtime/tmp"
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
  --env "PYTHONPATH=$workspace/lehome51-site:$campaign/external/lehome-challenge/source/lehome:$campaign/external/lehome-challenge:$campaign/src" \
  "${mask[@]}" \
  "$workspace/container/bhl.sif" "$workspace/venv/bin/python" -u \
  "$campaign/scripts/finetune_rasterised.py" \
  --policy_path "$source_checkpoint" \
  --capture_glob "$boundary_capture" \
  --boundary_capture "$boundary_capture" \
  --raster_replay_glob "$raster_replay_glob" \
  --raster_replay_files 4 --raster_replay_frames_per_file 8 \
  --heldout_glob "$heldout_glob" --heldout_files 4 --heldout_frames_per_file 8 \
  --task "fold the garment on the table" \
  --out "$out" --steps 300 --batch_size 5 --lr 1e-5 \
  --save_every 25 --unfreeze boundary --seed 4242
