#!/bin/bash
# One driver tick: advance whatever is unblocked and schedule the next tick.
# Runs on the host, not in the container, because it calls sbatch and sacct.
#SBATCH --job-name=lh-v5-tick
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00

set -euo pipefail
campaign=${1:?absolute campaign root}
exec /nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python -u \
  "$campaign/scripts/driver.py" --campaign "$campaign" tick
