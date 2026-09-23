#!/bin/bash
# Phase: collection_expansion. One predeclared manifest row per array element.
#SBATCH --job-name=lh-cl2-expand
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --constraint=a40|rtx8000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

set -euo pipefail
campaign=${1:?absolute campaign root}

task_index=${SLURM_ARRAY_TASK_ID:?array index}
source "$campaign/slurm/_common.sh"
lh_runtime "${task_index}_expand"
lh_isaac "$campaign/scripts/run_rollout_task.py" \
  --campaign "$campaign" --phase collection_expansion --index "$task_index" --label baseline
