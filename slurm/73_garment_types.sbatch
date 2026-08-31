#!/bin/bash
#SBATCH --job-name=lh-types
#SBATCH --account=eecs
#SBATCH --partition=share,gpu,ampere,dgxh
#SBATCH --constraint="a40|rtx8000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --array=0-11%6
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Matched-pose replays across ALL FOUR garment types, not just short tops.
#
# Every verified fold so far is a Top_Short. The challenge ships four classes
# with different fold criteria and different numbers of conditions, so a result
# on one class says little about the others. Episode ranges: Top_Short 0-249,
# Top_Long 250-499, Pant_Short 500-749, Pant_Long 750-999 (25 per garment
# instance, 10 instances per class).
#
# Also the first sweep to run with the wrist cameras parented to the gripper.

set -euo pipefail
W=/nfs/hpc/share/$USER/Humanoid_Lite
REPO=$W/lehome-fold-repro
EPISODES=(5 6 250 251 252 275 500 501 502 750 751 752)
EP=${EPISODES[$SLURM_ARRAY_TASK_ID]}

read -r G POSE < <(python3 - "$EP" <<'PY'
import json, sys
ep = int(sys.argv[1])
info = json.load(open("/nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-data/"
                      "Datasets/example/four_types_merged/meta/garment_info.json"))
g = list(info)[ep // 25]
p = info[g][str(ep % 25)]["object_initial_pose"]
print(g, ":".join(f"{x:.8f}" for x in p))
PY
)

export GARMENT="$G" MATCH_POSE="$POSE" MATCH_SCALE=0.45 SETTLE=60 STEPS=400
export REPLAY=$W/lehome-data/Datasets/example/four_types_merged/data/chunk-000/file-000.parquet
export REPLAY_EP=$EP RESULT_SUFFIX="_replay_ep${EP}"
echo "=== array $SLURM_ARRAY_TASK_ID -> episode $EP | garment $G ==="
exec bash "$REPO/slurm/70_policy_rollout.sbatch"
