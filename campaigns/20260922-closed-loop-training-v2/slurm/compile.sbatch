#!/bin/bash
# Compile one completed collection and enforce the preregistered signal gates.
# Fails closed: a partial collection never reaches the compiler, and a
# degenerate learning signal exits 4 rather than training on uniform weights.
#
#   compile.sbatch CAMPAIGN iteration1                  first draw, manifest rows
#   compile.sbatch CAMPAIGN expanded                    first draw + its expansion
#   compile.sbatch CAMPAIGN iterationK PLAN             later draw, plan rows
#   compile.sbatch CAMPAIGN iterationK-expanded PLAN    later draw + its expansion
#SBATCH --job-name=lh-cl2-compile
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00

set -euo pipefail
campaign=${1:?absolute campaign root}
scope=${2:-iteration1}
plan=${3:-}
source "$campaign/slurm/_common.sh"
lh_runtime "compile_${scope}"
R="$campaign/rollouts"

case "$scope" in
  iteration1)
    dirs=("$R/iteration1")
    expected=$(jq '.collection | length' "$campaign/manifest.json") ;;
  expanded)
    dirs=("$R/iteration1" "$R/expansion")
    expected=$(jq '(.collection | length) + (.collection_expansion | length)' "$campaign/manifest.json") ;;
  iteration*-expanded)
    k=${scope#iteration}; k=${k%-expanded}
    dirs=("$R/iteration$k" "$R/iteration$k-expansion")
    expected=$(jq '(.collection | length) + (.collection_expansion | length)' "${plan:?plan required}") ;;
  iteration*)
    dirs=("$R/$scope")
    expected=$(jq '.collection | length' "${plan:?plan required}") ;;
  *) echo "unknown scope $scope" >&2; exit 2 ;;
esac

found=$(find "${dirs[@]}" -name trajectory.npz -type f 2>/dev/null | wc -l)
if [ "$found" -ne "$expected" ]; then
  echo "expected $expected trajectory files in ${dirs[*]}, found $found" >&2
  exit 2
fi
globs=()
for d in "${dirs[@]}"; do globs+=(--trajectory-glob "$d/*/trajectory.npz"); done

lh_torch "$campaign/scripts/verify_sources.py" --campaign "$campaign" \
  --out "$campaign/audit/verify_sources_compile_${scope}.json"

lh_torch "$campaign/scripts/build_rollout_dataset.py" "${globs[@]}" \
  --manifest "$campaign/manifest.json" \
  --out "$campaign/datasets/${scope}.npz" \
  --provenance "$campaign/analysis/${scope}-dataset-provenance.json" \
  --csv "$campaign/analysis/${scope}-episodes.csv" \
  --gate-out "$campaign/audit/collection_gate_${scope}.json"
