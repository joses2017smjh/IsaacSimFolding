#!/bin/bash
# v7 launch gate for the two changed interfaces, before any production task:
# verify sources -> one short search row (the off-metadata large-garment P_B
# placement) with the new branch telemetry -> check the telemetry, the
# spawn, and the compiler's pure functions on the recorded attempts.
#SBATCH --job-name=lh-v7-smoke
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --constraint=a40|rtx8000
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -euo pipefail
campaign=${1:?absolute campaign root}
source "$campaign/slurm/_common.sh"
lh_runtime "smoke"
echo "=== [1/3] verify sources ==="
lh_torch "$campaign/scripts/verify_sources.py" --campaign "$campaign" --skip-checkpoint \
  --out "$campaign/audit/verify_sources_smoke.json"
echo "=== [2/3] short search row with branch telemetry ==="
rc=0
lh_isaac "$campaign/scripts/run_rollout_task.py" --campaign "$campaign" --phase recovery_smoke --index 0 --label smoke || rc=$?
echo "=== [3/3] telemetry, spawn and compiler checks (runner exit $rc) ==="
# runs even if the runner failed, so a raised spawn/code error is reported
lh_torch "$campaign/scripts/check_telemetry.py" --campaign "$campaign" --out "$campaign/audit/smoke_telemetry.json" || rc=${rc/#0/$?}
echo "=== SMOKE COMPLETE (exit $rc) ==="
exit "$rc"
