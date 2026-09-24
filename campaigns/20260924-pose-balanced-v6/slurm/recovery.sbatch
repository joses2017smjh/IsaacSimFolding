#!/bin/bash
# Bounded recovery search, one training-only row per array element: the H10
# student episode, root snapshots, and every preregistered candidate executed
# to a settled verdict. Every attempt is recorded.
#SBATCH --job-name=lh-v6-search
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --constraint=a40|rtx8000
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=02:55:00

set -euo pipefail
campaign=${1:?absolute campaign root}
task_index=${SLURM_ARRAY_TASK_ID:?array index}
source "$campaign/slurm/_common.sh"
lh_runtime "${task_index}_recovery"
lh_isaac "$campaign/scripts/run_rollout_task.py" \
  --campaign "$campaign" --phase recovery --index "$task_index" --label baseline
