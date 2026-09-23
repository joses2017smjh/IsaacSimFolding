#!/bin/bash
# Load a selected checkpoint exactly as the evaluator does and require one
# finite action and a config identical to the baseline's.
#
# Needs a GPU. The saved preprocessor pipeline pins device_processor to
# "cuda", and lerobot's get_safe_torch_device asserts CUDA is available, so a
# CPU-only reload fails with an AssertionError inside make_pre_post_processors
# (job 21400695). That is the evaluator's own loading path, so it is also the
# right thing to test: a checkpoint that only loads on CPU would not load for
# evaluation either.
#
#   reload.sbatch CAMPAIGN CHECKPOINT DATASET OUT
#SBATCH --job-name=lh-cl2-reload
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --constraint=a40|rtx8000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
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
