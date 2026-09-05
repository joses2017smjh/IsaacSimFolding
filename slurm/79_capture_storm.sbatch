#!/bin/bash
#SBATCH --job-name=lh-cap
#SBATCH --account=eecs
#SBATCH --partition=share,gpu,ampere,dgxh
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --array=0-15%4
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Capture the training set the domain gap says is missing.
#
# The policy scores +0.976 on path-traced frames and -0.038 on Storm frames of
# the same states -- a 1.014 gap, and the ONLY lever left after training,
# wrist cameras and mean-collapse were each measured and ruled out.
#
# Replay drives the recorded actions, so the action label is ground truth while
# the image is the distribution the policy is actually deployed on. That is
# exactly the pairing a rasterised fine-tune needs, and nothing else in this
# repo produces it -- the rollouts render these frames every step and throw
# them away.
#
# 16 episodes spanning all four garment classes.

set -euo pipefail
W=/nfs/hpc/share/$USER/Humanoid_Lite
REPO=$W/lehome-fold-repro
EPS=(2 4 5 6 25 26 27 28 251 252 275 276 500 501 750 752)
EP=${EPS[$SLURM_ARRAY_TASK_ID]}

read -r G POSE < <(python3 - "$EP" <<'PY'
import json, sys
ep=int(sys.argv[1])
info=json.load(open("/nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-data/"
                    "Datasets/example/four_types_merged/meta/garment_info.json"))
g=list(info)[ep//25]
p=info[g][str(ep%25)]["object_initial_pose"]
print(g, ":".join(f"{x:.8f}" for x in p))
PY
)

export GARMENT="$G" MATCH_POSE="$POSE" MATCH_SCALE=0.45 SETTLE=60 STEPS=400
export REPLAY=$W/lehome-data/Datasets/example/four_types_merged/data/chunk-000/file-000.parquet
export REPLAY_EP=$EP RESULT_SUFFIX="_cap_ep${EP}"
export CAPTURE_OUT=$W/lehome-data/storm_capture/ep${EP}.npz
echo "=== array $SLURM_ARRAY_TASK_ID -> episode $EP | garment $G ==="
exec bash "$REPO/slurm/70_policy_rollout.sbatch"
