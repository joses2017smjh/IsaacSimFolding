#!/bin/bash
#SBATCH --job-name=lh-sweep
#SBATCH --account=eecs
#SBATCH --partition=share,gpu,ampere,dgxh
#SBATCH --constraint="a40|rtx8000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Run the same demonstration against several garments and record what the
# OFFICIAL success checker says about each.
#
# The point is honest labelling. "Success" and "failure" here mean exactly what
# success_checker_garment_fold means -- not my impression of a frame. Every
# open-loop replay so far has scored False, and if that holds across garments
# then the repo has failure examples and no success example, which is the
# result and gets reported as such rather than dressed up.

set -euo pipefail
WORKSPACE=/nfs/hpc/share/$USER/Humanoid_Lite
REPO=$WORKSPACE/lehome-fold-repro
OUT=$REPO/results/outcome_sweep.tsv
: > "$OUT"
printf "garment\tsuccess\tdisplacement_m\tfell_m\n" >> "$OUT"

for G in Top_Short_Seen_0 Top_Long_Seen_0 Pant_Short_Seen_0 Pant_Long_Seen_0; do
    echo; echo "########## $G ##########"
    NPZ=$REPO/results/cloth_${G}.npz
    set +e
    GARMENT=$G STEPS=364 TRAJ=$REPO/assets/demo_episode0_action.npy \
      bash "$REPO/slurm/inner/_sweep_one.sh" "$G" "$NPZ" 2>&1 | tee /tmp/sweep_$G.log
    set -e
    S=$(grep -oE "^\[cloth\] SUCCESS=(True|False)" /tmp/sweep_$G.log | tail -1 | cut -d= -f2)
    D=$(grep -oE "TOTAL max displacement [0-9.]+" /tmp/sweep_$G.log | tail -1 | awk '{print $4}')
    F=$(grep -oE "fell [-0-9.]+ m" /tmp/sweep_$G.log | tail -1 | awk '{print $2}')
    printf "%s\t%s\t%s\t%s\n" "$G" "${S:-ERROR}" "${D:-}" "${F:-}" >> "$OUT"
done

echo; echo "=== outcome sweep ==="
cat "$OUT"
echo "=== done: $(date) ==="
