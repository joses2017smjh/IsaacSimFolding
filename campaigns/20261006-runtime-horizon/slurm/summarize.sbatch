#!/bin/bash
#SBATCH --job-name=lh-horizon-summary
#SBATCH --account=eecs
#SBATCH --partition=eecs,share
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
set -euo pipefail
campaign=${1:?campaign root}
/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python "$campaign/scripts/summarize.py" --campaign "$campaign" --write
