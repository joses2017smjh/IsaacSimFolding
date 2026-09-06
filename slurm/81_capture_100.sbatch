#!/bin/bash
#SBATCH --job-name=lh-cap100
#SBATCH --account=eecs
#SBATCH --partition=share,gpu,ampere,dgxh
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --array=0-99%6
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Scale the rasterised capture from 16 episodes to 100.
#
# 16 episodes of vision-only adaptation took the policy from inert to close:
# Storm-frame skill -0.038 -> +0.328, cloth displacement 14x, fold distances
# roughly halved, and one condition missed by 0.19 cm. Still 0 for 13. The two
# obvious levers are more data and unfreezing the action decoder; this is the
# first.
#
# 25 episodes from each garment class, evenly spaced, so no class dominates the
# way Top_Short did at 8 of 16.

set -euo pipefail
W=/nfs/hpc/share/$USER/Humanoid_Lite
REPO=$W/lehome-fold-repro
EPS=(0 10 21 31 42 52 62 73 83 93 104 114 124 135 145 156 166 176 187 197 208 218 228 239 249 250 260 271 281 292 302 312 323 333 343 354 364 374 385 395 406 416 426 437 447 458 468 478 489 499 500 510 521 531 542 552 562 573 583 593 604 614 624 635 645 656 666 676 687 697 708 718 728 739 749 750 760 771 781 792 802 812 823 833 843 854 864 874 885 895 906 916 926 937 947 958 968 978 989 999)
EP=${EPS[$SLURM_ARRAY_TASK_ID]}

read -r G POSE < <(python3 - "$EP" <<'PYEOF'
import json, sys
ep=int(sys.argv[1])
info=json.load(open("/nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-data/"
                    "Datasets/example/four_types_merged/meta/garment_info.json"))
g=list(info)[ep//25]
p=info[g][str(ep%25)]["object_initial_pose"]
print(g, ":".join(f"{x:.8f}" for x in p))
PYEOF
)

export GARMENT="$G" MATCH_POSE="$POSE" MATCH_SCALE=0.45 SETTLE=60 STEPS=400
export REPLAY=$W/lehome-data/Datasets/example/four_types_merged/data/chunk-000/file-000.parquet
export REPLAY_EP=$EP RESULT_SUFFIX="_cap100_ep${EP}"
export CAPTURE_OUT=$W/lehome-data/storm_capture100/ep${EP}.npz
echo "=== array $SLURM_ARRAY_TASK_ID -> episode $EP | garment $G ==="
exec bash "$REPO/slurm/70_policy_rollout.sbatch"
