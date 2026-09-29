#!/bin/bash
# One matched array for one GPU-model block. --matched-index maps even
# indices to the baseline and odd indices to the candidate on row N // 2, so
# a single array holds both policies and, throttled at %8, every concurrent
# batch runs four of each: same queue, same node pool, same hour.
# The driver pins the block's GPU model with --constraint on the sbatch
# command line (which overrides the directive below), and the task checks
# the model it actually got before and after Isaac runs.
#   args: campaign root, phase (remeasure), run label (= block), GPU model (a40|rtx8000)
#SBATCH --job-name=lh-v9-matched
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --constraint=a40|rtx8000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00

set -euo pipefail
campaign=${1:?absolute campaign root}
phase=${2:?remeasure}
run=${3:?run label}
gpu_model=${4:?a40|rtx8000}
index=${SLURM_ARRAY_TASK_ID:?array index}
source "$campaign/slurm/_common.sh"
lh_runtime "${phase}_${run}_${index}"
lh_isaac "$campaign/scripts/run_matched_task.py" \
  --campaign "$campaign" --phase "$phase" --matched-index "$index" --run "$run" --gpu-model "$gpu_model"
