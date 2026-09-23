#!/bin/bash
# Exact validated pose-3 action-5/action-10 roots; ordinary fresh H10 candidate
# branches. The cached H50 stream is a reconstruction control, never a label.
#SBATCH --job-name=lh-cl2-boundary
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --constraint=a40|rtx8000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00

set -euo pipefail
campaign=${1:?absolute campaign root}
checkpoint=${2:?checkpoint path}
label=${3:?checkpoint label}
source "$campaign/slurm/_common.sh"
lh_runtime "boundary_${label}"
lh_isaac "$campaign/scripts/run_boundary_evaluation.py" \
  --campaign "$campaign" --checkpoint "$checkpoint" --label "$label"
