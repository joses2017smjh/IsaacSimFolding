#!/bin/bash
# Compile validated recovery examples and apply the preregistered coverage
# gate. Exit 4 means the search did not supply enough coverage to train on.
#SBATCH --job-name=lh-v3-compile
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:40:00

set -euo pipefail
campaign=${1:?absolute campaign root}
source "$campaign/slurm/_common.sh"
lh_runtime "compile_recovery"
lh_torch "$campaign/scripts/verify_sources.py" --campaign "$campaign" \
  --out "$campaign/audit/verify_sources_compile_recovery.json"
lh_torch "$campaign/scripts/build_recovery_dataset.py" --campaign "$campaign" \
  --out "$campaign/datasets/recovery.npz" \
  --provenance "$campaign/analysis/recovery-dataset-provenance.json" \
  --gate-out "$campaign/audit/recovery_coverage_gate.json"
