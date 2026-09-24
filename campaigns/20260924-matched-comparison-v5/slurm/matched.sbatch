#!/bin/bash
# One matched array. --matched-index makes run_rollout_task.py map even
# indices to the baseline and odd indices to the candidate on row N // 2, so a
# single array holds both policies and, throttled at %8, every concurrent
# batch runs four of each: same queue, same node pool, same hour.
#   args: campaign root, phase (smoke|benchmark|final_test), run label (r1..)
#SBATCH --job-name=lh-v5-matched
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --constraint=a40|rtx8000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00

set -euo pipefail
campaign=${1:?absolute campaign root}
phase=${2:?smoke|benchmark|final_test}
run=${3:?run label}
index=${SLURM_ARRAY_TASK_ID:?array index}
source "$campaign/slurm/_common.sh"
lh_runtime "${phase}_${run}_${index}"
lh_isaac "$campaign/scripts/run_rollout_task.py" \
  --campaign "$campaign" --phase "$phase" --matched-index "$index" --run "$run"
