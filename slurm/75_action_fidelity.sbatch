#!/bin/bash
#SBATCH --job-name=lh-fid
#SBATCH --account=eecs
#SBATCH --partition=share,gpu,ampere,dgxh
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Measure the domain gap instead of invoking it.
#
# I have twice named the rasterised-vs-path-traced gap as the suspect for the
# policy's failure without ever quantifying it. This runs the policy on the
# ORIGINAL path-traced dataset frames -- fully in-distribution, no simulator,
# no Storm -- and scores its actions against the recorded ones, with a
# constant mean-action baseline so the number is interpretable.
#
# CKPT defaults to the 15K checkpoint every rollout so far has used. Point it
# at a later checkpoint to re-measure once the resume job (21124825) advances.

set -euo pipefail
export LH_ROUTE=${LH_ROUTE:-train}
source /nfs/hpc/share/$USER/Humanoid_Lite/lehome-fold-repro/slurm/_env.sh
setup_node_cache
CKPT=${CKPT:-$LEHOME_DATA/outputs/train/bc_smolvla_seed0/checkpoints/015000/pretrained_model}
TAG=${TAG:-015000}
echo "=== node: $(hostname) | ckpt $TAG | $(date) ==="
lh_exec bash -c "cd \"\$LEHOME\" && exec \"\$PY\" -u \
  $REPO/scripts/eval/action_fidelity.py \
  --policy_path '$CKPT' \
  --dataset_root \"\$LEHOME_DATA/Datasets/example/four_types_merged\" \
  --out '$REPO/results/action_fidelity_${TAG}.json'"
echo "=== done: $(date) ==="
