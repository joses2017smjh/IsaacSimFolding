#!/bin/bash
#SBATCH --job-name=lh-g0
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Gate G0: an episode runs end to end and the OFFICIAL scorer emits a verdict.
#
# Runs both floors (see scripts/g0_policies.py for why two). The numbers this
# produces are the denominator for G1, so they are recorded per garment type
# rather than averaged -- long pants and long-sleeved tops are not the same
# task as shorts, and a single number hides that from the first gate onward.
set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/lehome-fold-repro/slurm/_env.sh
setup_node_cache

export POLICY_TYPE=${POLICY_TYPE:-g0_uniform}
export GARMENT_TYPE=${GARMENT_TYPE:-top_long}
export NUM_EPISODES=${NUM_EPISODES:-2}
export MAX_STEPS=${MAX_STEPS:-600}
export SEED=${SEED:-0}
export OUT_CSV=$REPO/results/g0_floor.csv
mkdir -p "$REPO/results"

echo "=== node: $(hostname) | $POLICY_TYPE / $GARMENT_TYPE | $(date) ==="
lh_exec "$REPO/slurm/inner/g0_floor.sh"
echo "=== done: $(date) ==="
