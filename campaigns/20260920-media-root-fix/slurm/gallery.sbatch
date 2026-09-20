#!/bin/bash
#SBATCH --job-name=lh-media-report
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
set -euo pipefail
campaign=${1:?absolute campaign root}
exec python3 "$campaign/scripts/media_report.py" --campaign "$campaign"
