#!/bin/bash
# Load a selected checkpoint exactly as the evaluator does and require one
# finite action and a config identical to the baseline's. CPU only: this is a
# loadability gate, not a measurement, and should not spend GPU budget.
#
#   reload.sbatch CAMPAIGN CHECKPOINT DATASET OUT
#SBATCH --job-name=lh-cl2-reload
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00

set -euo pipefail
campaign=${1:?absolute campaign root}
checkpoint=${2:?checkpoint}
dataset=${3:?dataset}
out=${4:?output json}
source "$campaign/slurm/_common.sh"
lh_runtime "reload"
lh_torch "$campaign/scripts/check_checkpoint.py" \
  --checkpoint "$checkpoint" --dataset "$dataset" \
  --baseline "$(jq -r '.baseline_checkpoint.path' "$campaign/manifest.json")" \
  --out "$out"
