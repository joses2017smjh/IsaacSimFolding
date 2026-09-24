#!/bin/bash
# Compile the pose-balanced corpus: the per-pose coverage gate on the new
# search first (exit 4 = supervision still missing; nothing is trained), then
# v4's pinned corpus + the new examples with pose-balanced awr_weight.
#SBATCH --job-name=lh-v6-compile
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00

set -euo pipefail
campaign=${1:?absolute campaign root}
source "$campaign/slurm/_common.sh"
lh_runtime "compile"
lh_torch "$campaign/scripts/verify_sources.py" --campaign "$campaign" --skip-checkpoint \
  --out "$campaign/audit/verify_sources_compile.json"
lh_torch "$campaign/scripts/build_balanced_dataset.py" --campaign "$campaign" \
  --out "$campaign/datasets/recovery.npz" \
  --provenance "$campaign/analysis/recovery-dataset-provenance.json" \
  --gate-out "$campaign/audit/recovery_coverage_gate.json"
