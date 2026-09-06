#!/bin/bash
#SBATCH --job-name=lh-repro
#SBATCH --account=eecs
#SBATCH --partition=share,gpu,ampere,dgxh
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --array=0-7%4
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Is the first policy fold reproducible, or one lucky spawn pose?
#
# Pant_Short_Seen_0 folded at 5/5 with the checker firing at step 119. That is
# a single configuration. This runs the SAME class across eight different
# recorded spawn poses -- four more instances of Seen_0 and four of Seen_1 --
# so the answer is a rate rather than an anecdote.
#
# A single success at one pose and failure at seven would mean the policy found
# one configuration, not that it folds short pants.

set -euo pipefail
W=/nfs/hpc/share/$USER/Humanoid_Lite
REPO=$W/lehome-fold-repro
EPS=(500 503 506 509 512 515 518 521)
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
export CKPT=$W/lehome-data/outputs/train/bc_smolvla_raster_ft_full
export RESULT_SUFFIX="_repro_ep${EP}"
echo "=== array $SLURM_ARRAY_TASK_ID -> pose of episode $EP | garment $G | POLICY ==="
exec bash "$REPO/slurm/70_policy_rollout.sbatch"
