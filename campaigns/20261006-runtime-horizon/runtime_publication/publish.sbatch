#!/bin/bash
#SBATCH --job-name=lh-horizon-publish
#SBATCH --account=eecs
#SBATCH --partition=eecs,share
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python \
  /nfs/hpc/share/sanchej7/Computer_Vision/project_results_upgrade/folding/runtime_publication/publish_results.py \
  --publication-manifest-sha256 "${1:?publication manifest SHA256}"
