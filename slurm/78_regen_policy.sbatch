#!/bin/bash
#SBATCH --job-name=lh-rpol
#SBATCH --account=eecs
#SBATCH --partition=share,gpu,ampere,dgxh
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --array=0-3%2
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Re-record the four POLICY rollouts that rendered an empty table. Same seam
# fix; these are policy-driven rather than replay, so the verdict is expected
# to stay `failure` -- the policy is 0 for 8 and the renderer was never why.

set -euo pipefail
W=/nfs/hpc/share/$USER/Humanoid_Lite
REPO=$W/lehome-fold-repro
G=(Top_Long_Seen_0 Pant_Short_Seen_0 Pant_Short_Seen_1 Pant_Long_Seen_1)
export GARMENT=${G[$SLURM_ARRAY_TASK_ID]}
export STEPS=300 SETTLE=60 RESULT_SUFFIX="_v2policy"
echo "=== array $SLURM_ARRAY_TASK_ID -> $GARMENT (policy) ==="
exec bash "$REPO/slurm/70_policy_rollout.sbatch"
