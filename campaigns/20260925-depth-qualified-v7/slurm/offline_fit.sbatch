#!/bin/bash
# Offline fit diagnostic: predicted chunks vs validated recovery labels.
#   offline_fit.sbatch CAMPAIGN "label=path,label=path" OUT
#SBATCH --job-name=lh-v7-fit
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --constraint=a40|rtx8000
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
set -euo pipefail
campaign=${1:?}; ckpts=${2:?}; out=${3:?}
source "$campaign/slurm/_common.sh"
lh_runtime "offline_fit"
lh_torch "$campaign/scripts/offline_fit.py" --campaign "$campaign" --checkpoints "$ckpts" --out "$out"
