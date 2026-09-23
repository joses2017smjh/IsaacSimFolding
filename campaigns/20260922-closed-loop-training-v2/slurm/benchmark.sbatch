#!/bin/bash
# Phase: benchmark. One predeclared manifest row per array element.
#SBATCH --job-name=lh-cl2-bench
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --constraint=a40|rtx8000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00

set -euo pipefail
campaign=${1:?absolute campaign root}
checkpoint=${2:?checkpoint path}
label=${3:?checkpoint label}
task_index=${SLURM_ARRAY_TASK_ID:?array index}
source "$campaign/slurm/_common.sh"
lh_runtime "${task_index}_benchmark"
lh_isaac "$campaign/scripts/run_rollout_task.py" \
  --campaign "$campaign" --phase benchmark --index "$task_index" --checkpoint "$checkpoint" --label "$label"
