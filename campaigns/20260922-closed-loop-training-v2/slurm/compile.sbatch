#!/bin/bash
# Compile the completed collection and enforce the preregistered signal gates.
# Fails closed: a partial collection never reaches the compiler, and a
# degenerate learning signal exits 4 rather than training on uniform weights.
#SBATCH --job-name=lh-cl2-compile
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00

set -euo pipefail
campaign=${1:?absolute campaign root}
scope=${2:-iteration1}
source "$campaign/slurm/_common.sh"
lh_runtime "compile_${scope}"

if [ "$scope" = "expanded" ]; then
  glob="$campaign/rollouts/*/*/trajectory.npz"
  expected=$(jq '(.collection | length) + (.collection_expansion | length)' "$campaign/manifest.json")
  found=$(find "$campaign/rollouts/iteration1" "$campaign/rollouts/expansion" -name trajectory.npz -type f 2>/dev/null | wc -l)
else
  glob="$campaign/rollouts/iteration1/*/trajectory.npz"
  expected=$(jq '.collection | length' "$campaign/manifest.json")
  found=$(find "$campaign/rollouts/iteration1" -name trajectory.npz -type f 2>/dev/null | wc -l)
fi
if [ "$found" -ne "$expected" ]; then
  echo "expected $expected trajectory files, found $found" >&2
  exit 2
fi

lh_torch "$campaign/scripts/verify_sources.py" --campaign "$campaign" \
  --out "$campaign/audit/verify_sources_compile_${scope}.json"

lh_torch "$campaign/scripts/build_rollout_dataset.py" \
  --trajectory-glob "$glob" \
  --manifest "$campaign/manifest.json" \
  --out "$campaign/datasets/${scope}.npz" \
  --provenance "$campaign/analysis/${scope}-dataset-provenance.json" \
  --csv "$campaign/analysis/${scope}-episodes.csv" \
  --gate-out "$campaign/audit/collection_gate_${scope}.json"
