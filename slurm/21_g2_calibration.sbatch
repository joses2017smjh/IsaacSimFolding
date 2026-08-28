#!/bin/bash
#SBATCH --job-name=lh-g2
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Gate G2. Pure numpy on the predictions 20_train_value dumped -- no GPU, no
# simulator. Exit 0 pass, 2 fail, 3 could not evaluate.
set -euo pipefail
export LH_ROUTE=${LH_ROUTE:-train}
source /nfs/hpc/share/$USER/Humanoid_Lite/lehome-fold-repro/slurm/_env.sh
set +e
lh_exec "$REPO/slurm/inner/g2_calibration.sh"
rc=$?
set -e
case $rc in
  0) echo "G2 PASS" ;;
  2) echo "G2 FAIL -- do not start Stage 3; its advantages would be noise" ;;
  *) echo "G2 could not be evaluated (exit $rc)" ;;
esac
exit $rc
