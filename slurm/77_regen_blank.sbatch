#!/bin/bash
#SBATCH --job-name=lh-regen
#SBATCH --account=eecs
#SBATCH --partition=share,gpu,ampere,dgxh
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --array=0-4%3
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Re-record the episodes that rendered an empty table.
#
# Their physics and verdicts were always correct -- the garment mesh drew
# nothing because UV-seam duplicates made the render vertex count exceed the
# particle count. Fixed in storm_obs.py; these re-runs replace GIFs that show
# a caption over bare table.

set -euo pipefail
W=/nfs/hpc/share/$USER/Humanoid_Lite
REPO=$W/lehome-fold-repro
EPS=(250 251 252 501 502)
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
export REPLAY_EP=$EP RESULT_SUFFIX="_v2_ep${EP}"
echo "=== array $SLURM_ARRAY_TASK_ID -> episode $EP | garment $G ==="
exec bash "$REPO/slurm/70_policy_rollout.sbatch"
