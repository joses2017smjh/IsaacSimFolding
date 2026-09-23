#!/bin/bash
#SBATCH --job-name=lh-fold-analysis
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
campaign=${1:?campaign root}
mode=${2:-pilot}
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$campaign/analysis/cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"
if [ "$mode" = smoke ]; then
    input_root=/nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-fold-repro/campaigns/20260921-folding-pilot-v5
else
    input_root="$campaign"
fi
/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python "$campaign/analysis/analyze.py" \
    --root "$input_root" --mode "$mode" --output "$campaign/analysis/$mode-results"
