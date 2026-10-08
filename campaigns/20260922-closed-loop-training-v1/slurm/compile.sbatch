#!/bin/bash
# Compile only the completed predeclared collection; fails closed on missing or
# misaligned trajectories rather than silently training a partial corpus.
#SBATCH --job-name=lh-cl-v1-compile
#SBATCH --account=eecs
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail
campaign=${1:?absolute campaign root}
workspace=/nfs/hpc/share/sanchej7/Humanoid_Lite
repo="$workspace/lehome-fold-repro"
expected=$(jq '.collection | length' "$campaign/manifest.json")
actual=$(find "$campaign/rollouts/iteration1" -name trajectory.npz -type f | wc -l)
if [ "$actual" -ne "$expected" ]; then
  echo "expected $expected trajectory files, found $actual" >&2
  exit 2
fi
"$workspace/venv/bin/python" -u "$campaign/scripts/build_rollout_dataset.py" \
  --trajectory-glob "$campaign/rollouts/iteration1/*/trajectory.npz" \
  --out "$campaign/datasets/iteration1.npz" \
  --provenance "$campaign/analysis/iteration1-dataset-provenance.json" \
  --csv "$campaign/analysis/iteration1-episodes.csv"
