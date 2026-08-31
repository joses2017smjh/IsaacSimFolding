#!/bin/bash
#SBATCH --job-name=lh-mrep
#SBATCH --account=eecs
#SBATCH --partition=share,gpu,ampere,dgxh
#SBATCH --constraint="a40|rtx8000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --array=0-7%4
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Matched-pose demonstration replays, hunting for a full 5/5 success.
#
# Replaying episode 0 against a RANDOM spawn pose passed 3/5 conditions;
# replaying it against its OWN recorded pose passed 4/5, folding two axes
# outright (28.32 -> 3.87 and 37.96 -> 4.44 cm). The one that misses,
# dist(p1,p5) at 15.76 vs <=9.0, is consistent with open-loop drift: a fixed
# action sequence cannot correct simulation divergence. Different episodes
# will drift differently, so some should land 5/5.
#
# A success here is a DEMONSTRATION REPLAY verified by the challenge's own
# checker -- it is not the policy folding, and the GIF filename says so.

set -euo pipefail
W=/nfs/hpc/share/$USER/Humanoid_Lite
REPO=$W/lehome-fold-repro
EPISODES=(1 2 3 4 25 26 27 28)
EP=${EPISODES[$SLURM_ARRAY_TASK_ID]}

read -r G POSE STEPS < <(python3 - "$EP" <<'PY'
import json, sys
ep = int(sys.argv[1])
info = json.load(open("/nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-data/"
                      "Datasets/example/four_types_merged/meta/garment_info.json"))
ks = list(info)
g = ks[ep // 25]
p = info[g][str(ep % 25)]["object_initial_pose"]
print(g, ":".join(f"{x:.8f}" for x in p), 400)
PY
)

export GARMENT="$G" MATCH_POSE="$POSE" MATCH_SCALE=0.45 SETTLE=60
export REPLAY=$W/lehome-data/Datasets/example/four_types_merged/data/chunk-000/file-000.parquet
export REPLAY_EP=$EP RESULT_SUFFIX="_replay_ep${EP}" STEPS
echo "=== array $SLURM_ARRAY_TASK_ID -> episode $EP | garment $G | pose $POSE ==="
exec bash "$REPO/slurm/70_policy_rollout.sbatch"
