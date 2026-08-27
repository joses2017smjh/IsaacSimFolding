#!/bin/bash
#SBATCH --job-name=lh-rtx51
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# THE BLOCKING JOB. Nothing downstream is worth starting until this answers.
#
# LeHome pins isaacsim 5.1.0 and needs three RGB cameras. This cluster's 5.1
# RTX renderer segfaults inside omni.usd.create_hydra_engine -- that is the
# documented reason bhl-robustness-ladder has no RGB anywhere in it. If that
# still holds with LeHome's own render settings, the reproduction cannot run on
# the official stack and needs a stated deviation.
#
# PROBE_STACK selects which stack to probe: `borrowed` (default) needs
# NOTHING built and answers the question today; `official` / `source` re-run the
# same probe on the real stack once 01_install_stack has produced it.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/lehome-fold-repro/slurm/_env.sh

PROBE_STACK=${PROBE_STACK:-borrowed}
case "$PROBE_STACK" in
    borrowed)
        # The already-built BHL v51 stack: isaacsim 5.1.0.0 + isaaclab
        # 2.3.2.post1 + py3.11, versus LeHome's 5.1.0 + forked 2.3.1 + py3.11.
        # Good enough to answer the RENDERER question and it costs no install:
        # the Hydra engine comes from Kit inside `isaacsim`, not from isaaclab,
        # so the fork cannot change whether create_hydra_engine survives. What
        # it CANNOT tell you is whether LeHome's garment scene renders -- only
        # whether this cluster's 5.1 can open a render product at all.
        # Read-only: it imports from that venv and writes nothing into it.
        SIF=$WORKSPACE/container/bhl.sif
        PY=$WORKSPACE/venv/bin/python
        ;;
    official)
        # The organizers' image, once 01_install_stack has converted it.
        SIF=$WORKSPACE/container/lehome-official.sif
        PY=/opt/lehome-challenge/.venv/bin/python
        ;;
    source)
        SIF=$WORKSPACE/container/lehome.sif
        PY=$WORKSPACE/venv-lehome/bin/python
        ;;
    *) echo "unknown PROBE_STACK=$PROBE_STACK" >&2; exit 1 ;;
esac

setup_node_cache
export PROBE_OUT=$REPO/results/rtx51_probe_${PROBE_STACK}.txt
mkdir -p "$REPO/results"

echo "=== node: $(hostname) | stack: $PROBE_STACK | venv: $UV_PROJECT_ENVIRONMENT | $(date) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

# The probe is expected to be able to die hard. Do not let `set -e` swallow the
# exit status before it can be reported -- a segfault (139) is a RESULT here,
# not an accident, and it is the one this job exists to distinguish.
set +e
lh_exec "$REPO/slurm/inner/rtx_probe.sh"
rc=$?
set -e

echo "=== inner exit code: $rc ==="
case $rc in
    0)   echo "RESULT: RTX renders on 5.1 -- official stack is viable." ;;
    139) echo "RESULT: SEGFAULT (139). 5.1 renderer is broken here, as recorded." ;;
    *)   echo "RESULT: exit $rc -- read $PROBE_OUT for the last line reached." ;;
esac
echo "--- $PROBE_OUT ---"
cat "$PROBE_OUT" 2>/dev/null || echo "(no verdict file -- died before first emit)"
echo "=== done: $(date) ==="
