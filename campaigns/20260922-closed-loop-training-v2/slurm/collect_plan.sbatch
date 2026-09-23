#!/bin/bash
# Iteration >= 2 collection: one row of a resolved plan per array element,
# rolled out from the plan's collection checkpoint. Every row is retained.
#
#   collect_plan.sbatch CAMPAIGN PLAN {collection|collection_expansion} CHECKPOINT
#SBATCH --job-name=lh-cl2-collectk
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --constraint=a40|rtx8000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

set -euo pipefail
campaign=${1:?absolute campaign root}
plan=${2:?resolved plan}
phase=${3:?collection or collection_expansion}
checkpoint=${4:?collection checkpoint}
task_index=${SLURM_ARRAY_TASK_ID:?array index}
source "$campaign/slurm/_common.sh"
lh_runtime "${task_index}_collect_plan"
lh_isaac "$campaign/scripts/run_rollout_task.py" \
  --campaign "$campaign" --phase "$phase" --index "$task_index" \
  --plan "$plan" --checkpoint "$checkpoint" --label plan
